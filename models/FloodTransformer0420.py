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

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

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
        #print("h_prime",h_prime.shape)

        # 2. 准备 attention 输入，组合节点对特征
        # h_i: [B, N, N, num_heads, out_features]
        h_i = h_prime.unsqueeze(2).repeat(1, 1, N, 1, 1)
        h_j = h_prime.unsqueeze(1).repeat(1, N, 1, 1, 1)
        #print("h_i",h_i.shape)
        #print("h_j",h_j.shape)

        # 拼接特征对，得到 [B, N, N, num_heads, 2 * out_features]
        a_input = torch.cat([h_i, h_j], dim=-1)
        #print("a_input",a_input.shape)


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
def attention_fusion_concat(attn_outputs):
    # 多个 attention 输出拼接后线性映射
    fused = torch.cat(attn_outputs, dim=-1)  # (batch, node_num, total_embed_dim)
    projection = nn.Linear(fused.size(-1), attn_outputs[0].size(-1)).to(fused.device)
    return projection(fused)

def attention_fusion_gate(attn_outputs):
    # 使用门控机制融合
    gate_weights = nn.Parameter(torch.zeros(len(attn_outputs))).to(attn_outputs[0].device)
    gate_weights = torch.softmax(gate_weights, dim=0)  # softmax 归一化
    fused = sum(w * output for w, output in zip(gate_weights, attn_outputs))
    return fused


# === 总模型 ===
class Model(nn.Module):
    def __init__(self,configs):
        super().__init__()
        # self.feature_dim = configs.feature_dim
        # self.embed_dim = configs.embed_dim
        # self.num_heads = configs.num_heads
        self.feature_dim = None
        self.embed_dim = configs.d_model
        self.num_heads = configs.n_heads
        # 使用已有的 Transformer Encoder
        self.continuous_projectors = nn.ModuleList([
            nn.Linear(1, 64),
            nn.Linear(1, 64),
            nn.Linear(1, 64),
            nn.Linear(1, 64),
            nn.Linear(1, 64),
            nn.Linear(1, 64)
        ])
        # 类别 embedding，假设类别数量为 num_classes
        self.num_classes = 7
        self.class_embed_dim =64 #这里还要改参数
        self.class_embedding = nn.Embedding(self.num_classes, self.class_embed_dim)
        
        # Embedding
        self.enc_embedding_rain = DataEmbedding(448, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.enc_embedding_flood = DataEmbedding(448, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.rain_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
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
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
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
        
       

        
        # Projection to prediction
        self.projection4rain = nn.Linear(configs.d_model, configs.pred_len)
        self.projection4flood = nn.Linear(512, 24)
        self.fuse_to_output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 24),
            nn.ReLU()

        )
        self.fuse2_to_output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU()

        )
        # 可训练的残差权重参数（初始化为 1.0）
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.dec_proj = nn.Linear(24, 512)  # 把 dec_out 投影到 512
        self.W1 = nn.Linear(512, 512)
        self.W2 = nn.Linear(512, 512)


       

    def forward(self, x_enc,future_rain, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):

        #加载静态内容
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
        # bias_matrix= bias_matrix1 + bias_matrix2  # (1, 1, node_num, node_num)
        bias_matrix2 = bias_matrix2 * adj
        # 模型正式部分：
        # 1. 类别嵌入
        batch_size, seq_len, node_num, feature_dim = x_enc.shape
        #print("x_enc.shape:",x_enc.shape)
        # 假设类别标签是 feature_dim 的最后一维，取整数类别索引
        class_labels = x_enc[..., -1].long()  # (batch_size, seq_len, node_num)
        class_embeds = self.class_embedding(class_labels)  # (batch_size, node_num, seq_len, embed_dim)
        #print("类别嵌入class_embeds.shape:",class_embeds.shape)

        # 2. 连续特征映射
        # 连续特征在 feature 的前 6 个维度
        continuous_features = x_enc[..., :6]  # (B, T, N, 6)
        # 将每个连续特征分别投影（每个维度分别走一个 Linear）
        projected_features = []
        for i in range(6):
            # 取第 i 个连续特征，形状 (B, T, N, 1)
            feat = continuous_features[..., i:i+1]
            # 使用对应的投影器，Linear(1, 64)，输入形状为 (B, T, N, 1) → reshape 为 (B*T*N, 1)
            feat_proj = self.continuous_projectors[i](feat.reshape(-1, 1))  # (B*T*N, 64)
            feat_proj = feat_proj.reshape(batch_size, seq_len, node_num, -1)  # (B, T, N, 64)
            projected_features.append(feat_proj)

        # 拼接所有连续特征的嵌入：(B, T, N, 64*6=384)
        continuous_embeds = torch.cat(projected_features, dim=-1)

        # 拼接类别 embedding 到原始特征
        x_enc = torch.cat([continuous_embeds, class_embeds], dim=-1)  # (batch_size, node_num, seq_len, feature_dim -1 + embed_dim)
        #print("拼接类别 embedding 到原始特征x_enc.shape:",x_enc.shape)
        batch_size, node_num,seq_len, feature_dim = x_enc.shape


        # 2. 分离洪水和降雨数据，分别 Encoder
        rain_x =x_enc[:, :7,:, :]    # (batch, 7, seq_len, feature_dim)
        flood_x = x_enc[:, 7:,:, :]   # (batch, 56, seq_len, feature_dim)
        #print("分离洪水和降雨数据，分别 Encoder rain_x.shape:",rain_x.shape)
        # 调整维度用于 Encoder 输入 (batch, seq_len, nodes, feature) -> (batch * nodes, seq_len, feature)
        rain_x = rain_x.reshape(batch_size * 7, seq_len, feature_dim)
        flood_x = flood_x.reshape(batch_size * 56, seq_len, feature_dim)           
        enc_out_rain = self.enc_embedding_rain(rain_x,None)
        enc_out_flood = self.enc_embedding_flood(flood_x,None)


       

        # 取最后一个时间步
        rain_encoded = enc_out_rain[:, -1, :]  # (224(32*7), 512)
        flood_encoded = enc_out_flood[:, -1, :]  # (1792(32*56), 512)

        # reshape 回 batch 维度
        rain_encoded = rain_encoded.view(batch_size, 7, -1)  # (32, 7, 512)
        flood_encoded = flood_encoded.view(batch_size, 56, -1)  # (32, 56, 512)
        # 合并节点特征
        node_embeddings = torch.cat([rain_encoded, flood_encoded], dim=1)  # (32, 63, 512)
        #print("node_embeddings:",node_embeddings.shape)
        

        

        # 使用自定义的 GAT 层进行图传播
        # 1. 输入节点嵌入和邻接矩阵
        # 2. GAT 层内部会完成：线性变换 -> 注意力打分 -> Softmax 归一化 -> 特征聚合
        # bias=bias_matrix1
        #print("Bias stats:", bias.min(), bias.max(), torch.isnan(bias).any())
        # bias=bias_matrix2
        #print("Bias stats:", bias.min(), bias.max(), torch.isnan(bias).any())

        # 两个gat_layer待改进
        attn_output1 = self.gat_layer_location(node_embeddings, adj, bias_matrix1)  # (batch, node_num, embed_dim)
        attn_output2 = self.gat_layer_terrain(node_embeddings, adj, bias_matrix2)  # (batch, node_num, embed_dim)
        # 因果attention

        # 融合方式一：拼接
        h_out = attention_fusion_concat([attn_output1, attn_output2])
        #print("h_out:",h_out.shape)
        # 融合方式二：gate
        h_out_2 = attention_fusion_gate([attn_output1, attn_output2])
        #print("h_out_2:",h_out_2.shape)
        h_out=h_out_2
        dec_out = self.projection4rain(h_out)  # (batch, node_num, embed_dim) -> (batch, node_num, pred_len)
        # print("rain_pred:",dec_out.shape)[32, 63, 24] 这里有问题，会涉及到 63->7 的转换

        dec_out = F.relu(dec_out)
        # print("future_rain:",future_rain.shape)[32, 63, 24])

        # dec_out: [B, T, 24] -> project to [B, T, 512]
        dec_out_proj = self.dec_proj(dec_out)

        # 计算 gate
        gate = torch.sigmoid(self.W1(h_out) + self.W2(dec_out_proj))  # [B, T, 512]

        # 融合
        fused = gate * h_out + (1 - gate) * dec_out_proj  # [B, T, 512]

        h_proj = self.projection4flood(h_out)  # [B, T, 24]
        fused = self.alpha * h_proj + self.beta * dec_out
        # flood_pred = self.fuse_to_output(fused)
        # flood_pred = F.relu(flood_pred)
        
        flood_pred = self.fuse2_to_output(fused)

        #print("flood_pred:",flood_pred.shape)

        return flood_pred



# === 主程序 ===
if __name__ == "__main__":
    # 假数据
    batch_size = 2
    seq_len = 5
    node_num = 63
    feature_dim = 5
    embed_dim = 16
    num_heads = 4

    fake_features = torch.rand(batch_size, seq_len, node_num, feature_dim)
    fake_features[..., 2:4] *= 10.0
