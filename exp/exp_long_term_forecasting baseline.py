from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')

class MSE_MAELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.alpha = alpha  # 权重调节

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
        # data_set就是Dataset_Custom
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = MSE_MAELoss(alpha=0.5)  # 以 MSE 为主，兼顾 MAE
        return criterion
    
    def save_model_and_input(self, example_input, setting):
        """
        将模型保存为 TorchScript，并保存示例输入
        """
        # 构建模型输入示例
        batch_x = example_input
        
        print(example_input.shape)
        # 将模型转换为 TorchScript
        self.model.eval()
        batch_x=batch_x.to("cpu")  
        # 使用 torch.jit.trace 记录模型
        #traced_model = torch.jit.script(self.model.to("cpu"))
        traced_model = torch.jit.trace(self.model.to("cpu"), (batch_x))
        
        # 保存模型
        model_save_path = f'./checkpoints/{setting}/model_traced.pt'
        traced_model.save(model_save_path)
        print(f"✅ 模型已保存至 {model_save_path}")

        # 保存示例输入
        input_save_path = f'./checkpoints/{setting}/example_input.pt'
        torch.save(example_input, input_save_path)
        print(f"✅ 示例输入已保存至 {input_save_path}")

    def save_model_and_input(self, example_input, setting):
        """
        将模型保存为 TorchScript，并保存示例输入
        """
        # 确保 batch_x 和 future_rain_data 都包含在内
        batch_x, future_rain_data = example_input  # 假设 example_input 是一个包含 (batch_x, future_rain_data) 的元组

        print(batch_x.shape, future_rain_data.shape)
        
        # 将模型转换为 TorchScript
        self.model.eval()
        batch_x = batch_x.to("cpu")
        future_rain_data = future_rain_data.to("cpu")  # 确保 future_rain_data 在 CPU 上

        # 使用 torch.jit.trace 记录模型
        traced_model = torch.jit.trace(self.model.to("cpu"), (batch_x, future_rain_data))
        
        # 保存模型
        model_save_path = f'./checkpoints/{setting}/model_traced.pt'
        traced_model.save(model_save_path)
        print(f"✅ 模型已保存至 {model_save_path}")

        # 保存示例输入
        input_save_path = f'./checkpoints/{setting}/example_input.pt'
        torch.save(example_input, input_save_path)
        print(f"✅ 示例输入已保存至 {input_save_path}")

 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,future_rain_data) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp_known = batch_y[:, :, :self.args.label_len, :]                      # shape: [B, N, L_label, D]
                dec_inp_zeros = torch.zeros_like(batch_y[:, :, -self.args.pred_len:, :])   # shape: [B, N, L_pred, D]
                dec_inp = torch.cat([dec_inp_known, dec_inp_zeros], dim=2)       # shape: [B, N, L_label + L_pred, D]
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    B = batch_x.shape[0]  # 动态 batch 大小
                    batch_x = batch_x[:, 7:, :, 0:1].reshape(B * 56, 24, 1)
                    batch_x_mark=batch_x_mark.unsqueeze(1).repeat(1, 56, 1, 1).reshape(B * 56,24, 4)
                    dec_inp=dec_inp[:,7:,:,0:1].reshape(B * 56,48, 1)
                    batch_y_mark=batch_y_mark.unsqueeze(1).repeat(1, 56, 1, 1).reshape(B * 56,48, 4)

                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs=outputs.view(B, 56, -1, 1) 
                    batch_y=batch_y[:,7:, -self.args.pred_len:,0:1]

                

               
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

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,future_rain_data) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                dec_inp_known = batch_y[:, :, :self.args.label_len, :]                      # shape: [B, N, L_label, D]
                dec_inp_zeros = torch.zeros_like(batch_y[:, :, -self.args.pred_len:, :])   # shape: [B, N, L_pred, D]
                dec_inp = torch.cat([dec_inp_known, dec_inp_zeros], dim=2)       # shape: [B, N, L_label + L_pred, D]

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # 这里不走，走另外一条
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        # print("batch_y:",batch_y.shape)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    B = batch_x.shape[0]  # 动态 batch 大小
                    batch_x = batch_x[:, 7:, :, 0:1].reshape(B * 56, 24, 1)
                    batch_x_mark=batch_x_mark.unsqueeze(1).repeat(1, 56, 1, 1).reshape(B * 56,24, 4)
                    dec_inp=dec_inp[:,7:,:,0:1].reshape(B * 56,48, 1)
                    batch_y_mark=batch_y_mark.unsqueeze(1).repeat(1, 56, 1, 1).reshape(B * 56,48, 4)

                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs=outputs.view(B, 56, -1, 1) 
                    batch_y=batch_y[:,7:, -self.args.pred_len:,0:1]

                    


                    # 确保在 device 上
                    batch_y = batch_y.to(self.device)
                   
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        example_input = None  # 用于保存示例输入
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,future_rain_data) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp_known = batch_y[:, :, :self.args.label_len, :]                      # shape: [B, N, L_label, D]
                dec_inp_zeros = torch.zeros_like(batch_y[:, :, -self.args.pred_len:, :])   # shape: [B, N, L_pred, D]
                dec_inp = torch.cat([dec_inp_known, dec_inp_zeros], dim=2)       # shape: [B, N, L_label + L_pred, D]
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    B = batch_x.shape[0]  # 动态 batch 大小
                    batch_x = batch_x[:, 7:, :, 0:1].reshape(B * 56, 24, 1)
                    batch_x_mark=batch_x_mark.unsqueeze(1).repeat(1, 56, 1, 1).reshape(B * 56,24, 4)
                    dec_inp=dec_inp[:,7:,:,0:1].reshape(B * 56,48, 1)
                    batch_y_mark=batch_y_mark.unsqueeze(1).repeat(1, 56, 1, 1).reshape(B * 56,48, 4)

                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs=outputs.view(B, 56, -1, 1) 
                    batch_y=batch_y[:,7:, -self.args.pred_len:,0:1]
               

               
                # 确保在 device 上
                batch_y = batch_y.to(self.device)

                # 将当前 batch 保存为输入用例
                if example_input is None:  # 只保存第一个 batch
                    example_input = batch_x

                # 将数据转回 CPU 并保存
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # 可选：反归一化
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                

                preds.append(outputs)
                trues.append(batch_y)

                # if i % 5 == 0:
                #     # input = batch_x.detach().cpu().numpy()
                #     # if test_data.scale and self.args.inverse:
                #     #     shape = input.shape
                #     #     input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                #     # gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                #     # pd = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                #     # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                #     input = batch_x.detach().cpu().numpy()   # [32, 63, 24, 7]
                #     outputs = outputs.cpu().numpy() if torch.is_tensor(outputs) else outputs
                #     batch_y = batch_y.cpu().numpy() if torch.is_tensor(batch_y) else batch_y

                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)

                #     # 取 input 最后一个时间步的第一个特征（比如第0维）
                #     input_last = input[0, :, -1, 0]      # → [63]
                #     true_last = batch_y[0, :, -1]        # → [63]
                #     pred_last = outputs[0, :, -1]        # → [63]

                #     gt = np.concatenate((input_last, true_last), axis=0)  # → [126]
                #     pd = np.concatenate((input_last, pred_last), axis=0)  # → [126]

                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # 保存模型和示例输入
        #
        #  self.save_model_and_input(example_input, setting)
        #example_input = (batch_x, future_rain_data)  # 创建一个包含两个输入的元组
        # self.save_model_and_input(example_input, setting)

        # 合并预测和真实值
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # 结果保存
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
         # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
            dtw = "{:.4f}".format(dtw)  

        else:
            dtw = 'Not calculated'
        
        # trues = trues[:, :, :56] 
        # preds = preds[:, :, :56] 
        # 计算 NSE 和 KGE
        def nse(y_true, y_pred):
            return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

        def kge(y_true, y_pred):
            r = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
            alpha = np.std(y_pred) / np.std(y_true)
            beta = np.mean(y_pred) / np.mean(y_true)
            return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        nse_score = nse(trues, preds)
        kge_score = kge(trues, preds)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        print('mse:{:.4f}, mae:{:.4f}, nse:{:.4f},kge:{:.4f}'.format(mse, mae, nse_score,kge_score))
        
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))

        # 激增样本检测
        # def is_spike(sample, threshold=2.0):
        #     """
        #     判断一个样本是否包含激增。阈值单位：标准差的倍数。
        #     """
        #     diffs = np.diff(sample, axis=0)
        #     max_diff = np.max(np.abs(diffs))
        #     std = np.std(sample)
        #     return max_diff > threshold * std if std > 0 else False
        def is_spike(sample, threshold=2.0):
            """
            判断一个样本是否包含激增。阈值单位：标准差的倍数。
            """
            if len(sample) < 2:
                return False  # 太短无法判断激增
            diffs = np.diff(sample, axis=0)
            max_diff = np.max(np.abs(diffs))
            std = np.std(sample)
            return max_diff > threshold * std if std > 0 else False
        # Spike detection in test set
        spike_mae_list, spike_mse_list = [], []
        spike_nse_list, spike_kge_list = [], []
        spike_count = 0

        for i in range(preds.shape[0]):  # 遍历 batch
                true_seq = trues[i,:,0]  # [24]
                pred_seq = preds[i,:,0]  # [24]
                if is_spike(true_seq):
                    spike_count += 1
                    spike_mae_list.append(np.mean(np.abs(true_seq - pred_seq)))
                    spike_mse_list.append(np.mean((true_seq - pred_seq) ** 2))
                    spike_nse_list.append(nse(true_seq, pred_seq))
                    spike_kge_list.append(kge(true_seq, pred_seq))

        if spike_count > 0:
            print(f"📈 激增样本数量: {spike_count} / {preds.shape[0] * preds.shape[1]}")
            print(f"激增样本 MAE: {np.mean(spike_mae_list):.4f}")
            print(f"激增样本 MSE: {np.mean(spike_mse_list):.4f}")
            print(f"激增样本 NSE: {np.mean(spike_nse_list):.4f}")
            print(f"激增样本 KGE: {np.mean(spike_kge_list):.4f}")
        else:
            print("无激增样本。")

        f = open("result_long_term_forecast_baseline.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{:.4f}, mae:{:.4f}, nse:{:.4f},kge:{:.4f},spike_mse:{:.4f},spike_mae:{:.4f},spike_nse:{:.4f},spike_kge:{:.4f},'.format(mse, mae, nse_score,kge_score,np.mean(spike_mse_list),np.mean(spike_mae_list),np.mean(spike_nse_list),np.mean(spike_kge_list)))
        f.write('\n')
        f.write('\n')
        f.close()



        return
