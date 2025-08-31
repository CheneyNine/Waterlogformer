from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from torch.nn import Parameter
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
import matplotlib.pyplot as plt
class StaticGatingFusion(nn.Module):
    """
    Gate-based fusion of dynamic x and static s:
      gate = σ(Wg [x; s] + bg)
      out  = gate * x + (1 - gate) * Ws(s)
    """
    def __init__(self, d_model, static_d):
        super().__init__()
        self.gate_lin = nn.Linear(d_model + static_d, d_model)
        self.static_proj = nn.Linear(static_d, d_model)
    def forward(self, x, s):
        # x: [B*, T, d_model], s: [B*, T, static_d]
        cat = torch.cat([x, s], dim=-1)
        gate = torch.sigmoid(self.gate_lin(cat))
        s_proj = self.static_proj(s)
        return gate * x + (1 - gate) * s_proj

class EncoderLayerWithStatic(nn.Module):
    def __init__(self, attn_layer, d_model, d_ff, dropout, static_d):
        super().__init__()
        # reuse existing EncoderLayer internals
        self.enc_layer = EncoderLayer(attn_layer, d_model, d_ff, dropout)
        self.static_fusion = StaticGatingFusion(d_model, static_d)
    def forward(self, x, s, attn_mask=None):
        x, _ = self.enc_layer(x, attn_mask)
        x = self.static_fusion(x, s)
        return x, None

class DecoderLayerWithStatic(nn.Module):
    def __init__(self, self_attn, cross_attn, d_model, d_ff, dropout, static_d):
        super().__init__()
        # reuse existing DecoderLayer internals
        self.dec_layer = DecoderLayer(self_attn, cross_attn, d_model, d_ff, dropout)
        self.static_fusion = StaticGatingFusion(d_model, static_d)
    def forward(self, x, enc_out, s, self_mask=None, cross_mask=None):
        x = self.dec_layer(x, enc_out, self_mask, cross_mask)
        x = self.static_fusion(x, s)
        return x






# === 总模型 ===
class Model(nn.Module):
    def __init__(self,configs):
        super().__init__()
        # Load large static matrices once and register as buffers
        adj_np = np.load('/root/Time-Series-Library-main/dataset/adjacency_matrix.npy')
        self.register_buffer('adj_matrix', torch.tensor(adj_np, dtype=torch.float32))
        dist_np = np.load('/root/Time-Series-Library-main/dataset/distance_matrix_km.npy')
        self.register_buffer('distance_matrix', torch.tensor(dist_np, dtype=torch.float32))
        flow_np = np.load('/root/Time-Series-Library-main/dataset/flow_accum_weights.npy')
        self.register_buffer('flow_weights', torch.tensor(flow_np, dtype=torch.float32))
        slope_np = np.load('/root/Time-Series-Library-main/dataset/slope_correlation_matrix.npy')
        self.register_buffer('slope_corr_matrix', torch.tensor(slope_np, dtype=torch.float32))
        self.feature_dim = None
        self.embed_dim = configs.d_model
        self.num_heads = configs.n_heads
        self.use_cls_plus_dyn = True
        # 使用已有的 Transformer Encoder
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        # 动态特征 projector 和静态特征 MLP
        self.static_d = 128
        self.static_projectors = nn.ModuleList([
            nn.Linear(1, self.static_d) for _ in range(5)  # 静态连续值，共5个
        ])
        self.static_mlp = nn.Sequential(
            nn.Linear(self.static_d * 6, self.static_d),  # 降维
            nn.ReLU()
        )
        # 类别 embedding，假设类别数量为 num_classes
        self.num_classes = 7
        self.class_embedding = nn.Embedding(self.num_classes, 128)  # 类别静态值嵌入

        # 新增洪涝与静态信息融合MLP
        self.flood_static_fusion = nn.Sequential(
            nn.Linear(configs.d_model + self.static_d, configs.d_model),
            nn.ReLU()
        )
        
        # Embedding
        self.enc_embedding_rain = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.enc_embedding_flood = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,configs.dropout)
        self.dec_embedding_rain = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding_flood = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        # encoder layers with static fusion only for flood
        self.rain_encoder_layers = nn.ModuleList([
            EncoderLayerWithStatic(
                AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                               configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff, configs.dropout,
                static_d=self.static_d
            ) for _ in range(configs.e_layers)
        ])
        self.flood_encoder_layers = nn.ModuleList([
            EncoderLayerWithStatic(
                AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                               configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff, configs.dropout,
                static_d=self.static_d
            ) for _ in range(configs.e_layers)
        ])
        


        self.pred_len = configs.pred_len
        self.label_len = configs.label_len

        # decoder layers with static fusion for flood
        self.rain_decoder = Decoder(
            [  # keep original rain decoder
                DecoderLayer(
                    AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                   configs.d_model, configs.n_heads),
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                   configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        # flood decoder layers
        self.flood_decoder_layers = nn.ModuleList([
            DecoderLayerWithStatic(
                AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                               configs.d_model, configs.n_heads),
                AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                               configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff, configs.dropout,
                static_d=self.static_d
            ) for _ in range(configs.d_layers)
        ])
        self.flood_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

       

    def forward(self, x_enc, future_rain, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x_enc = x_enc[..., [0, 2, 3, 4, 5, 6, 7]]  # remove point_id at index 1
        x_dec = x_dec[..., [0]]  # remove point_id at index 1


        # 0.加载静态内容
        # Use preloaded adjacency buffer
        adj = self.adj_matrix.to(x_enc.device).unsqueeze(0).unsqueeze(1)
        # distance_matrix is available as self.distance_matrix if needed
       

        # 模型正式部分：
        # 1. 分离数据
        # x_enc: [batch_size, node_num, seq_len, feature_dim]
        batch_size, node_num, seq_len, feature_dim = x_enc.shape
        dynamic_feature = x_enc[..., 0:1]  # (B, N, T, 1), 动态特征
        static_continuous = x_enc[..., 1:6]  # (B, N, T, 5), 静态连续特征
        static_class = x_enc[..., 6].long()  # (B, N, T), 静态类别特征
        rain_input = dynamic_feature[:, :7, :, :]
        flood_rain_effect = self.compute_rainfall_influence(rain_input, adj)
        rain_input=rain_input.reshape(batch_size * 7, seq_len, 1)  # [B*7, T, 1]
        x_mark_enc_rain=x_mark_enc.unsqueeze(1).repeat(1, 7, 1, 1)
        x_mark_enc_rain= x_mark_enc_rain.reshape(batch_size * 7, seq_len, 4)
        
        
        x_mark_enc_future_rain=x_mark_dec[:,-self.pred_len:,:].unsqueeze(1).repeat(1, 7, 1, 1)
        x_mark_enc_future_rain= x_mark_enc_future_rain.reshape(batch_size * 7, seq_len, 4)

        flood_input = dynamic_feature[:, 7:, :, :]+flood_rain_effect
        flood_input=flood_input.reshape(batch_size * 56, seq_len, 1)  # [B*56, T, 1]
        
        # 2. 降雨数据嵌入、Encode
        # rain_input: [batch_size*7, seq_len, 1]
         # Embedding
        
        enc_out_rain = self.enc_embedding_rain(rain_input, x_mark_enc_rain)  # [batch_size*7, seq_len, d_model]
        # rain encoding with per-layer static fusion (static info for rain: use zeros)
        static_s_rain = torch.zeros_like(enc_out_rain[..., :self.static_d])
        xr = enc_out_rain
        for layer in self.rain_encoder_layers:
            xr, _ = layer(xr, static_s_rain)
        enc_out_rain = xr  # [batch_size*7, seq_len, d_model]

        # 3. 洪涝数据嵌入
        x_mark_enc_flood=x_mark_enc.unsqueeze(1).repeat(1, 56, 1, 1)
        x_mark_enc_flood= x_mark_enc_flood.reshape(batch_size * 56, seq_len, 4)

        flood_embedding = self.enc_embedding_flood(flood_input, x_mark_enc_flood)  # [batch_size*56, seq_len, d_model]

        # 4. 静态信息中嵌入
        projected_static = []
        for i in range(5):
            feat = static_continuous[..., i:i+1]  # [B, N, T, 1]
            feat_proj = self.static_projectors[i](feat.reshape(-1,1))  # [B*N*T, static_d]
            feat_proj = feat_proj.reshape(batch_size, node_num, seq_len, -1)  # [B, N, T, static_d]
            projected_static.append(feat_proj)

        class_embeds = self.class_embedding(static_class).reshape(batch_size, node_num, seq_len, -1)  # [B, N, T, static_d]
        static_embeds = torch.cat(projected_static + [class_embeds], dim=-1)  # [B, N, T, static_d*6]
        static_info_embedding = self.static_mlp(static_embeds)  # [B, N, T, static_d]

        flood_embedding = flood_embedding.reshape(batch_size, 56, seq_len, -1)  # [B, 56, T, d_model]
        static_info_embedding = static_info_embedding[:, 7:, :, :]  # [B, 56, T, static_d]
        # 5. 静态信息作为CLS token插入flood序列最前端
        # 构造CLS token: 静态信息+flood第一个时间步的embedding（只用于shape对齐）
        cls_token = self.flood_static_fusion(torch.cat([
            flood_embedding[:, :, :1, :],  # [B, 56, 1, d_model]
            static_info_embedding[:, :, :1, :]  # [B, 56, 1, static_d]
        ], dim=-1))  # [B, 56, 1, d_model]
        # flood_embedding: [B, 56, T, d_model]
        flood_token_seq = torch.cat([cls_token, flood_embedding], dim=2)  # [B, 56, T+1, d_model]
        # flood encoding with per-layer static fusion
        flood_x = flood_token_seq.reshape(batch_size * 56, seq_len + 1, -1)
        # prepare static sequence [B*56, T+1, static_d] (pad or repeat as needed)
        static_s_full = torch.cat([
            torch.zeros(batch_size, 56, 1, self.static_d, device=x_enc.device),
            static_info_embedding
        ], dim=2).reshape(batch_size * 56, seq_len + 1, self.static_d)
        xf = flood_x
        for layer in self.flood_encoder_layers:
            xf, _ = layer(xf, static_s_full)
        enc_out_flood = xf  # [B*56, T+1, d_model]
        # 保留全部时间步或仅最后一步
        rain_encoded = enc_out_rain.view(batch_size, 7, seq_len, -1)  # [B, 7, T, d_model]
        # 洪水flood encoder输出的CLS token和flood编码
        flood_cls_token = enc_out_flood[:, 0:1, :]  # [B*56, 1, d_model]
        flood_encoded = enc_out_flood[:, 1:, :].view(batch_size, 56, seq_len, -1)  # [B, 56, T, d_model]
    


       
 
      
        # 预测部分
        # 预测未来降雨
        x_dec_rain=x_dec[:, :7, :, :].reshape(batch_size * 7, self.pred_len+self.label_len, 1)
        x_dec_flood=x_dec[:, 7:, :, :].reshape(batch_size * 56, self.pred_len+self.label_len, 1)

        x_mark_dec_rain=x_mark_dec.unsqueeze(1).repeat(1, 7, 1, 1)
        x_mark_dec_rain= x_mark_dec_rain.reshape(batch_size * 7, self.pred_len+self.label_len, 4)

        x_mark_dec_flood=x_mark_dec.unsqueeze(1).repeat(1, 56, 1, 1)
        x_mark_dec_flood= x_mark_dec_flood.reshape(batch_size * 56, self.pred_len+self.label_len, 4)

        dec_out_rain = self.dec_embedding_rain(x_dec_rain, x_mark_dec_rain)
        dec_out_flood = self.dec_embedding_flood(x_dec_flood, x_mark_dec_flood)
        dec_out_rain = self.rain_decoder(dec_out_rain, enc_out_rain, x_mask=None, cross_mask=None)
        dec_out_rain=dec_out_rain[:,-self.pred_len:,:]
        rain_pred = dec_out_rain.view(batch_size, 7, self.pred_len)  # [B, 7, pred_len]



        # 洪水预测部分
        # 1. 对 rain_pred 使用与历史相同的 DataEmbedding
        #flood_rain_effect = self.compute_rainfall_influence(rain_pred, adj)
        # future_rain_input = rain_pred.reshape(batch_size * 7, self.pred_len, 1)  # [B*7, pred_len, 1]
        # future_rain_emb = self.enc_embedding_rain(future_rain_input, x_mark_enc_future_rain)  # [B*7, pred_len, d_model]
        # # 2. 将降雨节点嵌入扩展为洪水节点数
        # future_rain_emb = future_rain_emb.view(batch_size, 7, self.pred_len, -1)  # [B, 7, pred_len, d_model]
        # flood_rain_effect = self.compute_rainfall_influence(future_rain_emb, adj) # [4, 56, 24, 512]       


       

        last_dyn = flood_rain_effect[:, :, -1, :].contiguous().view(batch_size * 56, -1)
        # cls_plus_dyn = flood_cls_token.squeeze(1) + last_dyn  # [B*56, d_model]
        cls_plus_dyn = last_dyn   # [B*56, d_model]
        flood_context = cls_plus_dyn.unsqueeze(1).expand(-1, self.pred_len, -1)  # [B*56, pred_len, d_model]

        # 4. 利用 flood encoder 的 CLS token作为 cross-attention 的 key/value
        # flood_cls_token: [B*56, 1, d_model]，需扩展到 [B*56, pred_len, d_model]
        flood_encoded= flood_encoded.reshape(batch_size * 56, self.pred_len, -1)
        
        # flood decoding with static fusion
        dec_f = dec_out_flood  # [B*56, label_len+pred_len, d_model]
        # Prepare static info for decoder: static features per node (time-invariant)
        # static_info_embedding: [B, 56, seq_len, static_d]
        # Compute node-level static representation by averaging over encoder time dimension
        static_node_feat = static_info_embedding.mean(dim=2)  # [B, 56, static_d]
        # Expand to decoder sequence length
        dec_len = dec_f.size(1)
        static_s_dec = static_node_feat.unsqueeze(2).expand(batch_size, 56, dec_len, self.static_d)
        static_s_dec = static_s_dec.reshape(batch_size * 56, dec_len, self.static_d)  # [B*56, dec_len, static_d]
        for layer in self.flood_decoder_layers:
            dec_f = layer(dec_f, enc_out_flood, static_s_dec, self_mask=None, cross_mask=None)
        dec_out_flood = self.flood_projection(dec_f[:, -self.pred_len:, :])
        flood_pred = dec_out_flood.view(batch_size, 56, self.pred_len)

        # 5. 返回结果
        # rain_pred: [B, 7, pred_len], flood_pred: [B, 56, pred_len]
        pred_all = torch.cat([rain_pred, flood_pred], dim=1)  # [B, 63, pred_len]

       
        # 对比学习 contrastive losses
        reprs = flood_pred.unsqueeze(-1)  # [B, 56, T, 1]
        # Use preloaded distance_matrix buffer
        distance_matrix = self.distance_matrix.to(x_enc.device)
        near_mask = distance_matrix[7:, 7:] < 5.0

        tpi = x_enc[..., -1].long()
        # print(tpi.shape)
        # same_tpi_mask = (tpi[:, 7:] == tpi[:, 7:].unsqueeze(2)).bool()
        tpi_flood = tpi[:, 7:, :]  # [B, 56, T]
        same_tpi_mask = (tpi_flood.unsqueeze(2) == tpi_flood.unsqueeze(1))  # [B, 56, 56, T]
        same_tpi_mask = same_tpi_mask.all(dim=-1)  # 沿时间维度取逻辑与，输出 [B, 56, 56]

        strong_corr_mask = (self.slope_corr_matrix.to(x_enc.device)[-56:, -56:] > 0)

        contrastive_losses = []
        for t in range(seq_len):
            for b in range(batch_size):
                r = reprs[b, :, t, :]  # [56, d_model]
                contrastive_losses.append(self.contrastive_loss(r, near_mask))
                contrastive_losses.append(self.contrastive_loss(r, same_tpi_mask[b]))
                contrastive_losses.append(self.contrastive_loss(r, strong_corr_mask))
        total_contrastive_loss = sum(contrastive_losses) / len(contrastive_losses)

        return pred_all, total_contrastive_loss

    def attention_fusion_concat(self, attn_outputs):
        # 多个 attention 输出拼接后线性映射
        fused = torch.cat(attn_outputs, dim=-1)
        return self.concat_projection(fused)

    def attention_fusion_gate(self, attn_outputs):
        gate = torch.softmax(self.gate_weights, dim=0)
        fused = sum(w * output for w, output in zip(gate, attn_outputs))
        return fused



    def contrastive_loss(self, reprs, pos_mask, temperature=0.07):
        sim = F.cosine_similarity(reprs.unsqueeze(1), reprs.unsqueeze(0), dim=-1)  # [N, N]
        sim = sim / temperature
        log_prob = sim - sim.logsumexp(dim=1, keepdim=True)
        loss = -log_prob[pos_mask].mean()
        return loss
    def compute_rainfall_influence(self, rain_encoded, adj_matrix, alpha=None, beta=None):
        """
        输入:
            rain_encoded: Tensor, shape [B, 7, T, d_model] 降雨特征
            adj_matrix: Tensor, shape [1,1, 63, 63] 邻接矩阵
        返回:
            flood_rain_effect: Tensor, shape [B, 56, T, d_model]
        """
        B, _, T, D = rain_encoded.shape
        device = rain_encoded.device
        adj_matrix = adj_matrix[0, 0, :7, -56:]  # 取前7行，最后56列
        # print("adj_matrix:",adj_matrix.shape)
        # print("rain_encoded:",rain_encoded.shape)

        rain_to_flood = torch.einsum('sj,bsld->bjld', adj_matrix, rain_encoded)  # [B, 56, T, D]

        # 初始化参数
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta

        # Use preloaded flow_weights buffer
        flow_weights = self.flow_weights.to(device)[-56:, -56:]
        # 动态传播计算
        flood_rain_effect = []
        H_prev = torch.zeros(B, 56, D, device=device)
        for t in range(T):
            R_t = rain_to_flood[:, :, t, :]  # 当前时间步的降雨影响
            flow_acc = torch.einsum("ij,bjd->bid", flow_weights, H_prev)  # 上游积水汇聚
            H_t = alpha * R_t + beta * flow_acc
            flood_rain_effect.append(H_t.unsqueeze(2))
            H_prev = H_t

        return torch.cat(flood_rain_effect, dim=2)  # [B, 56, T, D]