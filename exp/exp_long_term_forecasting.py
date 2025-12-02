from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import torch.nn.functional as F
import time
import warnings
import numpy as np
from utils.dtw_metric import accelerated_dtw

warnings.filterwarnings('ignore')

class MSE_MAELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.alpha = 0.02

    def forward(self, pred, target):
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * self.mae(pred, target)

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = MSE_MAELoss(alpha=0.5)
        return criterion

    def save_model_and_input(self, example_input, setting):
        batch_x, future_rain_data, batch_x_mark, dec_inp, batch_y_mark = example_input
        self.model.eval()
        batch_x = batch_x.to("cpu")
        future_rain_data = future_rain_data.to("cpu")
        batch_x_mark = batch_x_mark.to("cpu")
        dec_inp = dec_inp.to("cpu")
        batch_y_mark = batch_y_mark.to("cpu")
        traced_model = torch.jit.trace(self.model.to("cpu"), (batch_x, future_rain_data, batch_x_mark, dec_inp, batch_y_mark))
        model_save_path = f'./checkpoints/{setting}/model_traced.pt'
        traced_model.save(model_save_path)
        input_save_path = f'./checkpoints/{setting}/example_input.pt'
        torch.save(example_input, input_save_path)

    def _single_mask_contrastive_loss(self, sim_matrix, pos_mask, temperature):
        sim = sim_matrix / temperature
        sim = sim - sim.logsumexp(dim=1, keepdim=True)
        log_prob = sim
        loss = -log_prob[pos_mask].mean()
        return loss

    def contrastive_loss(self, reprs, near_mask, same_tpi_mask, epoch, max_epochs=15, temperature=0.07):
        N, D = reprs.shape
        temp = temperature
        sim_matrix = F.cosine_similarity(reprs.unsqueeze(1), reprs.unsqueeze(0), dim=-1)
        alpha = 1.0 - min(epoch / (max_epochs * 0.3), 1.0)
        near_mask = near_mask.bool().to(reprs.device)
        same_tpi_mask = same_tpi_mask.bool().to(reprs.device)
        loss_near = self._single_mask_contrastive_loss(sim_matrix, near_mask, temp)
        loss_tpi = self._single_mask_contrastive_loss(sim_matrix, same_tpi_mask, temp)
        total_loss = alpha * loss_near + (1 - alpha) * loss_tpi
        return total_loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, future_rain_data) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp_known = batch_y[:, :, :self.args.label_len, :]
                dec_inp_zeros = torch.zeros_like(batch_y[:, :, -self.args.pred_len:, :])
                dec_inp_rand = torch.rand_like(batch_y[:, :, -self.args.pred_len:, :])
                dec_inp = torch.cat([dec_inp_known, dec_inp_rand], dim=2)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    mean_slice = batch_x[:, :7].mean(dim=1, keepdim=True)
                    remaining = batch_x[:, 7:]
                    batch_x = torch.cat((mean_slice, remaining), dim=1)
                    mean_slice = future_rain_data[:, :7].mean(dim=1, keepdim=True)
                    remaining = future_rain_data[:, 7:]
                    future_rain_data = torch.cat((mean_slice, remaining), dim=1)
                    mean_slice = dec_inp[:, :7].mean(dim=1, keepdim=True)
                    remaining = dec_inp[:, 7:]
                    dec_inp = torch.cat((mean_slice, remaining), dim=1)
                    outputs, contrastive_loss = self.model(batch_x, future_rain_data, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = 0
                batch_y = batch_y[:, :, -self.args.pred_len:, f_dim].to(self.device)
                batch_y = batch_y[:, 7:, :]
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        mse = nn.MSELoss()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, future_rain_data) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp_known = batch_y[:, :, :self.args.label_len, :]
                dec_inp_rand = torch.rand_like(batch_y[:, :, -self.args.pred_len:, :])
                dec_inp_zeros = torch.zeros_like(batch_y[:, :, -self.args.pred_len:, :])
                dec_inp = torch.cat([dec_inp_known, dec_inp_rand], dim=2)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    mean_slice = batch_x[:, :7].mean(dim=1, keepdim=True)
                    remaining = batch_x[:, 7:]
                    batch_x = torch.cat((mean_slice, remaining), dim=1)
                    mean_slice = future_rain_data[:, :7].mean(dim=1, keepdim=True)
                    remaining = future_rain_data[:, 7:]
                    future_rain_data = torch.cat((mean_slice, remaining), dim=1)
                    mean_slice = dec_inp[:, :7].mean(dim=1, keepdim=True)
                    remaining = dec_inp[:, 7:]
                    dec_inp = torch.cat((mean_slice, remaining), dim=1)
                    outputs, reprs = self.model(batch_x, future_rain_data, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = 0
                    batch_y = batch_y[:, :, -self.args.pred_len:, f_dim]
                    batch_y = batch_y[:, 7:, :]
                    batch_y = batch_y.to(self.device)
                    loss_sharpness = mse((outputs[:, :, 1:] - outputs[:, :, :-1]), (batch_y[:, :, 1:] - batch_y[:, :, :-1]))
                    distance_matrix = np.load('./dataset/distance_matrix_km.npy')
                    distance_matrix = torch.tensor(distance_matrix, dtype=torch.float32).to(batch_x.device)
                    slope_corr_matrix = np.load('./dataset/slope_correlation_matrix.npy')
                    slope_corr_matrix = torch.tensor(slope_corr_matrix, dtype=torch.float32).to(batch_x.device)
                    distance_matrix = distance_matrix.to(batch_x.device)
                    near_mask = distance_matrix[7:, 7:] < 5.0
                    tpi = batch_x[..., -1].long()
                    tpi_flood = tpi[:, 1:, :]
                    same_tpi_mask = (tpi_flood.unsqueeze(2) == tpi_flood.unsqueeze(1))
                    same_tpi_mask = same_tpi_mask.all(dim=-1)
                    strong_corr_mask = (slope_corr_matrix.to(batch_x.device)[-56:, -56:] > 0)
                    batch_size, node_num, seq_len, feature_dim = reprs.shape
                    contrastive_losses = []
                    for t in range(seq_len):
                        for b in range(batch_size):
                            r = reprs[b, :, t, :]
                            contrastive_losses.append(self.contrastive_loss(r, near_mask, same_tpi_mask[b], epoch))
                    total_contrastive_loss = sum(contrastive_losses) / len(contrastive_losses)
                    loss = criterion(outputs, batch_y) + self.args.lambda_contrast * total_contrastive_loss + loss_sharpness * self.args.sharpness
                    train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    iter_count = 0
                    time_now = time.time()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        example_input = None
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, future_rain_data) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp_known = batch_y[:, :, :self.args.label_len, :]
                dec_inp_rand = torch.rand_like(batch_y[:, :, -self.args.pred_len:, :])
                dec_inp_zeros = torch.zeros_like(batch_y[:, :, -self.args.pred_len:, :])
                dec_inp = torch.cat([dec_inp_known, dec_inp_rand], dim=2)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    mean_slice = batch_x[:, :7].mean(dim=1, keepdim=True)
                    remaining = batch_x[:, 7:]
                    batch_x = torch.cat((mean_slice, remaining), dim=1)
                    mean_slice = future_rain_data[:, :7].mean(dim=1, keepdim=True)
                    remaining = future_rain_data[:, 7:]
                    future_rain_data = torch.cat((mean_slice, remaining), dim=1)
                    mean_slice = dec_inp[:, :7].mean(dim=1, keepdim=True)
                    remaining = dec_inp[:, 7:]
                    dec_inp = torch.cat((mean_slice, remaining), dim=1)
                    outputs, contrastive_loss = self.model(batch_x, future_rain_data, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = 0
                batch_y = batch_y[:, :, -self.args.pred_len:, f_dim]
                batch_y = batch_y[:, 7:, :]
                batch_y = batch_y.to(self.device)
                if example_input is None:
                    example_input = batch_x
                outputs_np = outputs.detach().cpu().numpy()
                batch_y_np = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y_np.shape
                    if outputs_np.shape[-1] != batch_y_np.shape[-1]:
                        outputs_np = np.tile(outputs_np, [1, 1, int(batch_y_np.shape[-1] / outputs_np.shape[-1])])
                    outputs_np = test_data.inverse_transform(outputs_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y_np = test_data.inverse_transform(batch_y_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                outputs_np = outputs_np[:, :, f_dim:]
                batch_y_np = batch_y_np[:, :, f_dim:]
                preds.append(outputs_np)
                trues.append(batch_y_np)
                if i % 5 == 0:
                    input_np = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_np.shape
                        input_np = test_data.inverse_transform(input_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    input_last = input_np[0, :, -1, 0]
                    true_last = batch_y_np[0, :, -1]
                    pred_last = outputs_np[0, :, -1]
                    gt = np.concatenate((input_last, true_last), axis=0)
                    pd = np.concatenate((input_last, pred_last), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        example_input = (batch_x, future_rain_data, batch_x_mark, dec_inp, batch_y_mark)
        self.save_model_and_input(example_input, setting)
        batch_x_np = batch_x.cpu().numpy()
        future_rain_data_np = future_rain_data.cpu().numpy()
        np.save("batch_x.npy", batch_x_np)
        np.save("future_rain_data.npy", future_rain_data_np)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
            dtw = "{:.4f}".format(dtw)
        else:
            dtw = 'Not calculated'
        preds = preds[:, 7:, :]
        trues = trues[:, 7:, :]
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        def nse(y_true, y_pred):
            return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        def kge(y_true, y_pred):
            r = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
            alpha = np.std(y_pred) / np.std(y_true)
            beta = np.mean(y_pred) / np.mean(y_true)
            return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        nse_score = nse(trues, preds)
        kge_score = kge(trues, preds)
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, NSE: {}, KGE: {}'.format(mse, mae, nse_score, kge_score))
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.close()
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, nse_score, kge_score]))
        def is_spike(sample, threshold=2.0):
            diffs = np.diff(sample, axis=0)
            max_diff = np.max(np.abs(diffs))
            std = np.std(sample)
            return max_diff > threshold * std if std > 0 else False
        spike_mae_list, spike_mse_list = [], []
        spike_nse_list, spike_kge_list = [], []
        spike_count = 0
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                true_seq = trues[i, j, :]
                pred_seq = preds[i, j, :]
                if is_spike(true_seq):
                    spike_count += 1
                    spike_mae_list.append(np.mean(np.abs(true_seq - pred_seq)))
                    spike_mse_list.append(np.mean((true_seq - pred_seq) ** 2))
                    spike_nse_list.append(nse(true_seq, pred_seq))
                    spike_kge_list.append(kge(true_seq, pred_seq))
        return
