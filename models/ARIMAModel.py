import torch
import torch.nn as nn
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

    

from statsmodels.tsa.arima.model import ARIMA
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.order = (1, 1, 0)  # 或从 args 获取固定值，如 args.arima_order
        self.output_len = args.pred_len
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)


    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x: [batch, seq_len, input_dim] => 只支持 input_dim=1
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        input_dim = x.shape[2]
        
        preds = []
        for i in range(batch_size):
            series = x[i, :, 0].detach().cpu().numpy()  # 取单变量序列
            model = ARIMA(series, order=self.order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=self.output_len)
            preds.append(forecast)

        preds = torch.tensor(preds, dtype=torch.float32).unsqueeze(-1).to(x.device)  # [batch, pred_len, 1]
        return preds