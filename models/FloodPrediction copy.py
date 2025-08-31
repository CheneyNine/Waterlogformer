import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding,PositionalEmbedding
import torch.nn.functional as F 
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
class RainExpander(nn.Module):
    def __init__(self, in_dim=7, out_dim=56, d_model=32, n_layers=3):
        super(RainExpander, self).__init__()
        
        # 将时间维度扩展
        self.upscale = nn.Linear(in_dim, out_dim)
        
        # MLP 特征增强
        self.layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.LayerNorm(d_model)
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):

        # 保持 batch 维度不变，将时间维度扩展
        batch_size, vars, num_patches, patch_len = x.shape
        x = x.permute(0, 2, 3, 1)  # [32, 3, 16, 7]

        x = self.upscale(x)  # [32, 3, 16, 56]
        # print("xxx1:",x.shape)
        x = x.permute(0, 3, 1, 2)  # [32, 56, 3, 16]
        # print("xxx2:",x.shape)

        # 特征增强
        # x = x.reshape(-1, x.shape[2], x.shape[3])  # [32*56, 3, 16]
        # x = self.layers(x)
        # x = x.reshape(batch_size, -1, num_patches, patch_len)  # [32, 56, 3, 16]

        return x

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # 分离维度
        self.flood_dim = 1  # 洪水点位
        self.rain_dim = 1   # 降雨数据

        # 洪水数据的 patch embedding
        self.patch_embedding_flood = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.patch_embedding_rain = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)


        self.rain_expander = RainExpander(in_dim=1, out_dim=1, d_model=32, n_layers=3)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        
        self.head = FlattenHead(
            self.flood_dim, self.head_nf, configs.pred_len, head_dropout=configs.dropout
        )
        # 值嵌入：将每个patch映射到d_model维空间
        self.value_embedding = nn.Linear(patch_len, configs.d_model, bias=False)

        # 位置嵌入：为每个patch添加位置信息
        self.position_embedding = PositionalEmbedding(configs.d_model)

        self.dropout = nn.Dropout(configs.dropout)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # 分离洪水和降雨数据
        x_flood = x_enc[:, :, :self.flood_dim]   # 洪水数据
        x_rain = x_enc[:, :, self.flood_dim:]     # 降雨数据

        # 标准化洪水数据
        means = x_flood.mean(1, keepdim=True).detach()
        x_flood = x_flood - means
        stdev = torch.sqrt(torch.var(x_flood, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_flood /= stdev

        # 计算降雨数据的总和
        rain_sum = x_rain.sum()  # 按 batch 计算每个样本的总降雨量
        mask = (rain_sum >= 1).float().unsqueeze(1).unsqueeze(2)  # 如果降雨总和大于等于 1，则返回 1；否则 0

        # 如果降雨数据为0，直接返回0
        zero_out = torch.zeros(x_flood.shape[0], self.pred_len, x_flood.shape[-1]).to(x_flood.device)
        
        # 正常模型推理路径
        x_flood = x_flood.permute(0, 2, 1)
        enc_out_flood, n_vars_flood = self.patch_embedding_flood(x_flood)

        x_rain = x_rain.permute(0, 2, 1)
        enc_out_rain, n_vars_rain = self.patch_embedding_rain(x_rain)
        
        enc_out_rain_expanded = self.rain_expander(enc_out_rain)

        # 合并降雨特征与洪水数据
        x = enc_out_flood + enc_out_rain_expanded
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # 特征嵌入
        x = self.value_embedding(x)
        x = self.dropout(x + self.position_embedding(x))

        # Encoder
        enc_out, attns = self.encoder(x)

        enc_out = torch.reshape(enc_out, (-1, n_vars_flood, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # 反标准化
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # ReLU 处理（去负值）
        dec_out = F.relu(dec_out)

        # 使用 mask 动态选择：降雨为 0 时返回全零，否则返回正常预测结果
        final_out = mask * dec_out + (1.0 - mask) * zero_out

        return final_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  
    