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



# === 自定义GAT层 ===
class GraphAttentionLayer(nn.Module):
    """
    多头图注意力层，支持多头并行计算。
    """
    def __init__(self, in_features, out_features, num_heads=1, dropout=0.0, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)

        # 每个 head 的权重矩阵和 attention 参数
        self.W = Parameter(torch.Tensor(num_heads, in_features, out_features))
        self.a = Parameter(torch.Tensor(num_heads, 2 * out_features, 1))

        # 初始化参数
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, adj, bias_matrix):
        """
        h: [B, N, in_features]  batch 内每个样本节点特征
        adj: [N, N]             所有 batch 共享的邻接矩阵
        """
        B, N, _ = h.size()

        # 1. 线性变换，得到 [B, N, num_heads, out_features]
        h_prime = torch.einsum('bni,hio->bnho', h, self.W)

        # 2. 准备 attention 输入，组合节点对特征
        # h_i: [B, N, N, num_heads, out_features]
        h_i = h_prime.unsqueeze(2).repeat(1, 1, N, 1, 1)
        h_j = h_prime.unsqueeze(1).repeat(1, N, 1, 1, 1)

        # 拼接特征对，得到 [B, N, N, num_heads, 2 * out_features]
        a_input = torch.cat([h_i, h_j], dim=-1)


        # 3. 计算注意力 e_ij: [B, N, N, num_heads]
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        e = self.leakyrelu(torch.einsum('bijhf,hfo->bijh', a_input, self.a))# [B, N, N, H]
        # print("e:",e.shape)
          # 添加上 bias
        # bias_matrix: [N, N]->[B, N, N, H]
        # print("bias_matrix:",bias_matrix.shape)

        # 原始 bias_matrix: [1, 1, 63, 63]
        # 目标：变成 [B, 63, 63, num_heads]
        bias_matrix = bias_matrix.expand(B, -1, -1, -1)  # [B, 1, 63, 63]
        bias_matrix = bias_matrix.permute(0, 2, 3, 1)    # [B, 63, 63, 1]
        bias_matrix = bias_matrix.expand(-1, -1, -1, self.num_heads)  # [B, 63, 63, H]

        e = e + bias_matrix

        # 构造 mask
        B, N, _, H = e.shape
        # 确保 adj 是 [N, N]
        if adj.dim() != 2:
            adj = adj.squeeze()  # 把多余维度去掉

        mask = adj.unsqueeze(0).unsqueeze(-1)       # [1, N, N, 1]
        mask = mask.expand(B, N, N, H).bool()       # [B, N, N, H]

        # 应用 mask
        e_masked = e.masked_fill(~mask, float('-inf'))

        # softmax 注意力
        attention = F.softmax(e_masked, dim=2)         # [B, N, N, H]
        attention = self.dropout(attention)
        #print("attention:",attention.shape)

        # h_prime: [B, N, H, F]
        h_out = torch.einsum('bijh,bjhf->bihf', attention, h_prime)

        # # 4. mask 非邻居节点，adj shape: [N, N] -> [1, N, N, 1]
        # mask = adj.unsqueeze(0).unsqueeze(-1).bool()  # broadcast to [B, N, N, num_heads]
        # e_masked = e.masked_fill(~mask, float('-inf'))

        # # 5. softmax attention
        # attention = F.softmax(e_masked, dim=2)  # 对 dim=2 归一化

        # # 6. dropout
        # attention = self.dropout(attention)
        # #print("attention:",attention.shape)
        # # 7. 加权求和：attention: [B, N, N, num_heads], h_prime: [B, N, num_heads, out_features]
        # # 先调换维度以便矩阵相乘
        # h_out = torch.einsum('bnnh,bnho->bnho', attention, h_prime)

        # 8. 合并多头
        if self.concat:
            # 拼接所有 head: [B, N, num_heads * out_features]
            h_out = h_out.reshape(B, N, -1)
        else:
            # 平均所有 head: [B, N, out_features]
            h_out = h_out.mean(dim=2)

        return h_out


# === 总模型 ===
class Model(nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.feature_dim = None
        self.embed_dim = configs.d_model
        self.num_heads = configs.n_heads
        self.keep_all_timesteps = True
        self.use_cls_plus_dyn = True
        # 使用已有的 Transformer Encoder
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
        self.enc_embedding_rain = DataEmbedding(1, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.enc_embedding_flood = DataEmbedding(1, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.rain_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.flood_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        

        # 多头 GAT 层：支持多头和 bias 扩展
        self.gat_layer_location = GraphAttentionLayer(self.embed_dim, self.embed_dim, num_heads=self.num_heads, dropout=configs.dropout, concat=False)
        self.gat_layer_terrain = GraphAttentionLayer(self.embed_dim, self.embed_dim, num_heads=self.num_heads, dropout=configs.dropout, concat=False)
        
        
        # # Projection to prediction
        # self.projection4rain = nn.Linear(configs.d_model, configs.pred_len)
        # self.projection4flood = nn.Linear(512, 24)
        # self.fuse_to_output = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(512, 24),
        #     nn.ReLU()

        # )
        # self.fuse2_to_output = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(24, 24),
        #     nn.ReLU()

        # )
        # # 可训练的残差权重参数（初始化为 1.0）
        # self.alpha = nn.Parameter(torch.tensor(1.0))
        # self.beta = nn.Parameter(torch.tensor(1.0))
        # self.dec_proj = nn.Linear(24, 512)  # 把 dec_out 投影到 512
        # self.W1 = nn.Linear(512, 512)
        # self.W2 = nn.Linear(512, 512)

        # # 融合 projection 和 gate 权重
        self.concat_projection = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.gate_weights = nn.Parameter(torch.zeros(2))

        self.pred_len = configs.pred_len
        self.rain_decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        # 洪水预测用的 decoder
        self.flood_decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

       

    def forward(self, x_enc, future_rain, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):

        # 0.加载静态内容
        adj=np.load('/root/Time-Series-Library-main/dataset/adjacency_matrix.npy')
        adj = torch.tensor(adj, dtype=torch.float32, device=x_enc.device)  # (node_num, node_num)
        adj = adj.unsqueeze(0).unsqueeze(1)  # (1,1, node_num, node_num)
        #print("adj.shape:",adj.shape)
        # 加载 bias 矩阵
        bias_matrix1 = np.load('/root/Time-Series-Library-main/dataset/shortest_distance.npy')
        bias_matrix2 = np.load('/root/Time-Series-Library-main/dataset/slope_correlation_matrix.npy')
        bias_matrix2 = np.nan_to_num(bias_matrix2, nan=0.0, posinf=1e4, neginf=-1e4)    
        bias_matrix1 = torch.tensor(bias_matrix1, dtype=torch.float32, device=x_enc.device)  # (node_num, node_num)
        bias_matrix2 = torch.tensor(bias_matrix2, dtype=torch.float32, device=x_enc.device)  # (node_num, node_num)
        bias_matrix1 = bias_matrix1.unsqueeze(0).unsqueeze(1)  # (1, 1, node_num, node_num)
        bias_matrix2 = bias_matrix2.unsqueeze(0).unsqueeze(1)  # (1, 1, node_num, node_num)
        bias_matrix2 = bias_matrix2 * adj

        # 模型正式部分：
        # 1. 分离数据
        # x_enc: [batch_size, node_num, seq_len, feature_dim]
        batch_size, node_num, seq_len, feature_dim = x_enc.shape
        dynamic_feature = x_enc[..., 0:1]  # (B, N, T, 1), 动态特征
        static_continuous = x_enc[..., 1:6]  # (B, N, T, 5), 静态连续特征
        static_class = x_enc[..., 6].long()  # (B, N, T), 静态类别特征
        rain_input = dynamic_feature[:, :7, :, :].reshape(batch_size * 7, seq_len, 1)  # [B*7, T, 1]
        flood_input = dynamic_feature[:, 7:, :, :].reshape(batch_size * 56, seq_len, 1)  # [B*56, T, 1]
        
        # 2. 降雨数据嵌入、Encode
        # rain_input: [batch_size*7, seq_len, 1]
        enc_out_rain = self.enc_embedding_rain(rain_input, None)  # [batch_size*7, seq_len, d_model]
        enc_out_rain, _ = self.rain_encoder(enc_out_rain)  # [batch_size*7, seq_len, d_model]

        

        # 3. 洪涝数据嵌入
        flood_embedding = self.enc_embedding_flood(flood_input, None)  # [batch_size*56, seq_len, d_model]

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
        flood_x = flood_token_seq.reshape(batch_size * 56, seq_len + 1, -1)  # [B*56, T+1, d_model]
        enc_out_flood, _ = self.flood_encoder(flood_x)  # [B*56, T+1, d_model]
        # 保留全部时间步或仅最后一步
        if self.keep_all_timesteps:
            rain_encoded = enc_out_rain.view(batch_size, 7, seq_len, -1)  # [B, 7, T, d_model]
            # 洪水flood encoder输出的CLS token和flood编码
            flood_cls_token = enc_out_flood[:, 0:1, :]  # [B*56, 1, d_model]
            flood_encoded = enc_out_flood[:, 1:, :].view(batch_size, 56, seq_len, -1)  # [B, 56, T, d_model]
        else:
            rain_encoded = enc_out_rain[:, -1:, :].view(batch_size, 7, 1, -1)     # [B, 7, 1, D]
            flood_cls_token = enc_out_flood[:, 0:1, :]  # [B*56, 1, d_model]
            # flood只取最后一个时间步（去掉CLS），即最后一行
            flood_encoded = enc_out_flood[:, -1:, :].view(batch_size, 56, 1, -1)  # [B, 56, 1, D]
        node_embeddings = torch.cat([rain_encoded, flood_encoded], dim=1)  # [B, 63, T, d_model]

        # GAT 层按时间步独立处理
        # gat_outputs_time = []
        # for t in range(seq_len):
        #     node_t = node_embeddings[:, :, t, :]  # [B, 63, d_model]
        #     attn_t1 = self.gat_layer_location(node_t, adj, bias_matrix1)  # [B, 63, d_model]
        #     attn_t2 = self.gat_layer_terrain(node_t, adj, bias_matrix2)  # [B, 63, d_model]
        #     h_t = self.attention_fusion_gate([attn_t1, attn_t2])  # [B, 63, d_model]
        #     gat_outputs_time.append(h_t.unsqueeze(2))  # [B, 63, 1, d_model]

        if self.keep_all_timesteps:
            gat_outputs_time = []
            for t in range(seq_len):
                node_t = node_embeddings[:, :, t, :]  # [B, 63, D]
                attn_t1 = self.gat_layer_location(node_t, adj, bias_matrix1)
                attn_t2 = self.gat_layer_terrain(node_t, adj, bias_matrix2)
                h_t = self.attention_fusion_gate([attn_t1, attn_t2])  # [B, 63, D]
                gat_outputs_time.append(h_t.unsqueeze(2))  # [B, 63, 1, D]
            h_out = torch.cat(gat_outputs_time, dim=2)  # [B, 63, T, D]
        else:
            node_t = node_embeddings[:, :, 0, :]  # [B, 63, D]
            attn_t1 = self.gat_layer_location(node_t, adj, bias_matrix1)
            attn_t2 = self.gat_layer_terrain(node_t, adj, bias_matrix2)
            h_out = self.attention_fusion_gate([attn_t1, attn_t2]).unsqueeze(2)  # [B, 63, 1, D]
            

        # h_out shape: [batch_size, 63, seq_len, d_model]

        # 预测部分
        # 预测未来降雨
        dec_input = torch.zeros(batch_size * 7, self.pred_len, self.embed_dim, device=x_enc.device)  # [B*7, pred_len, d_model]
        rain_dec_out = self.rain_decoder(dec_input, enc_out_rain, x_mask=None, cross_mask=None)  # [B*7, pred_len]
        rain_pred = rain_dec_out.view(batch_size, 7, self.pred_len)  # [B, 7, pred_len]

        # 洪水预测部分
        

        # 1. 对 rain_pred 使用与历史相同的 DataEmbedding
        future_rain_input = rain_pred.reshape(batch_size * 7, self.pred_len, 1)  # [B*7, pred_len, 1]
        future_rain_emb = self.enc_embedding_rain(future_rain_input, None)  # [B*7, pred_len, d_model]
        # 2. 将降雨节点嵌入扩展为洪水节点数
        future_rain_emb = future_rain_emb.view(batch_size, 7, self.pred_len, -1)  # [B, 7, pred_len, d_model]
        adj_matrix = adj[0, 0, :7, 7:]  # [7, 56]
        # [B, 7, pred_len, d_model] -> [B, 56, pred_len, d_model]
        future_rain_for_flood = torch.einsum('sj,bsld->bjld', adj_matrix, future_rain_emb)  # [B, 56, pred_len, d_model]
        future_rain_for_flood = future_rain_for_flood.reshape(batch_size * 56, self.pred_len, -1)  # [B*56, pred_len, d_model]
        # 3. 获取融合后特征中的洪水节点
        if self.keep_all_timesteps:
            h_out_flood = h_out[:, 7:, :, :]  # [B, 56, pred_len, d_model]
        else:
            h_out_flood = h_out[:, 7:, -1:, :]  # [B, 56, 1, d_model]
            h_out_flood = h_out_flood.expand(-1, -1, self.pred_len, -1)  # broadcast to [B, 56, pred_len, d_model]
        h_out_flood = h_out_flood.contiguous().view(batch_size * 56, self.pred_len, -1)  # [B*56, pred_len, d_model]
        if self.use_cls_plus_dyn:
            # [B*56, d_model]
            last_dyn = h_out[:, 7:, -1, :].contiguous().view(batch_size * 56, -1)
            cls_plus_dyn = flood_cls_token.squeeze(1) + last_dyn  # [B*56, d_model]
            flood_context = cls_plus_dyn.unsqueeze(1).expand(-1, self.pred_len, -1)  # [B*56, pred_len, d_model]
        else:
            flood_context = h_out_flood + future_rain_for_flood  # 默认原来的逻辑
        # 4. 利用 flood encoder 的 CLS token作为 cross-attention 的 key/value
        # flood_cls_token: [B*56, 1, d_model]，需扩展到 [B*56, pred_len, d_model]
        flood_dec_input = torch.zeros(batch_size * 56, self.pred_len, self.embed_dim, device=x_enc.device)  # [B*56, pred_len, d_model]
        flood_dec_out = self.flood_decoder(
            flood_dec_input,
            flood_context,
            x_mask=None, cross_mask=None
        )  # [B*56, pred_len]
        flood_pred = flood_dec_out.view(batch_size, 56, self.pred_len)  # [B, 56, pred_len]
        # 5. 返回结果
        # rain_pred: [B, 7, pred_len], flood_pred: [B, 56, pred_len]
        pred_all = torch.cat([rain_pred, flood_pred], dim=1)  # [B, 63, pred_len]
        return pred_all

    def attention_fusion_concat(self, attn_outputs):
        # 多个 attention 输出拼接后线性映射
        fused = torch.cat(attn_outputs, dim=-1)
        return self.concat_projection(fused)

    def attention_fusion_gate(self, attn_outputs):
        gate = torch.softmax(self.gate_weights, dim=0)
        fused = sum(w * output for w, output in zip(gate, attn_outputs))
        return fused


