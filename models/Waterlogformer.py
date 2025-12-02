from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from torch.nn import Parameter
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class StaticGatingFusion(nn.Module):
    def __init__(self, d_model, static_d):
        super().__init__()
        self.gate_lin = nn.Linear(d_model + static_d, d_model)
        self.static_proj = nn.Linear(static_d, d_model)
    def forward(self, x, s):
        cat = torch.cat([x, s], dim=-1)
        gate = torch.sigmoid(self.gate_lin(cat))
        s_proj = self.static_proj(s)
        return gate * x + (1 - gate) * s_proj

class StaticFusionWithOutGate(nn.Module):
    def __init__(self, d_model, static_d):
        super().__init__()
        self.static_proj = nn.Linear(static_d, d_model)
    def forward(self, x, s):
        s_proj = self.static_proj(s)
        return x + s_proj

class CrossTimeAttentionDecoderLayer(nn.Module):
    def __init__(self, self_attn, cross_attn_hist, cross_attn_future, d_model, d_ff, dropout, activation="relu"):
        super().__init__()
        self.self_attn_layer = self_attn
        self.cross_attn_hist = cross_attn_hist
        self.cross_attn_future = cross_attn_future
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    def forward(self, x, enc_out_hist, enc_out_future_rain, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attn_layer(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attn_hist(x, enc_out_hist, enc_out_hist, attn_mask=cross_mask, tau=tau, delta=delta)[0])
        x = self.norm2(x)
        x = x + self.dropout(self.cross_attn_future(x, enc_out_future_rain, enc_out_future_rain, attn_mask=cross_mask, tau=tau, delta=delta)[0])
        y = x = self.norm3(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm4(x + y)

class DecoderLayerWithStaticWithFutureRain(nn.Module):
    def __init__(self, self_attn, cross_attn, cross_attn_future, d_model, d_ff, dropout, static_d):
        super().__init__()
        self.dec_layer = CrossTimeAttentionDecoderLayer(self_attn, cross_attn, cross_attn_future, d_model, d_ff, dropout)
        self.static_fusion = StaticGatingFusion(d_model, static_d)
    def forward(self, x, enc_out, s, enc_out_future_rain, self_mask=None, cross_mask=None):
        x = self.dec_layer(x, enc_out, enc_out_future_rain, self_mask, cross_mask)
        x = self.static_fusion(x, s)
        return x

class DecoderLayerWithStaticWithFutureRainNoGates(nn.Module):
    def __init__(self, self_attn, cross_attn, cross_attn_future, d_model, d_ff, dropout, static_d):
        super().__init__()
        self.dec_layer = CrossTimeAttentionDecoderLayer(self_attn, cross_attn, cross_attn_future, d_model, d_ff, dropout)
        self.static_fusion = StaticFusionWithOutGate(d_model, static_d)
    def forward(self, x, enc_out, s, enc_out_future_rain, self_mask=None, cross_mask=None):
        x = self.dec_layer(x, enc_out, enc_out_future_rain, self_mask, cross_mask)
        x = self.static_fusion(x, s)
        return x

class EncoderLayerWithStatic(nn.Module):
    def __init__(self, attn_layer, d_model, d_ff, dropout, static_d):
        super().__init__()
        self.enc_layer = EncoderLayer(attn_layer, d_model, d_ff, dropout)
        self.static_fusion = StaticGatingFusion(d_model, static_d)
    def forward(self, x, s, attn_mask=None):
        x, _ = self.enc_layer(x, attn_mask)
        x = self.static_fusion(x, s)
        return x, None

class EncoderLayerWithStaticNogate(nn.Module):
    def __init__(self, attn_layer, d_model, d_ff, dropout, static_d):
        super().__init__()
        self.enc_layer = EncoderLayer(attn_layer, d_model, d_ff, dropout)
        self.static_fusion = StaticFusionWithOutGate(d_model, static_d)
    def forward(self, x, s, attn_mask=None):
        x, _ = self.enc_layer(x, attn_mask)
        x = self.static_fusion(x, s)
        return x, None

class DecoderLayerWithStatic(nn.Module):
    def __init__(self, self_attn, cross_attn, d_model, d_ff, dropout, static_d):
        super().__init__()
        self.dec_layer = DecoderLayer(self_attn, cross_attn, d_model, d_ff, dropout)
        self.static_fusion = StaticGatingFusion(d_model, static_d)
    def forward(self, x, enc_out, s, self_mask=None, cross_mask=None):
        x = self.dec_layer(x, enc_out, self_mask, cross_mask)
        x = self.static_fusion(x, s)
        return x

class DecoderLayerWithStaticNoGate(nn.Module):
    def __init__(self, self_attn, cross_attn, d_model, d_ff, dropout, static_d):
        super().__init__()
        self.dec_layer = DecoderLayer(self_attn, cross_attn, d_model, d_ff, dropout)
        self.static_fusion = StaticFusionWithOutGate(d_model, static_d)
    def forward(self, x, enc_out, s, self_mask=None, cross_mask=None):
        x = self.dec_layer(x, enc_out, self_mask, cross_mask)
        x = self.static_fusion(x, s)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.without_static = configs.without_static
        self.without_rain = configs.without_rain
        self.without_gate = configs.without_gate
        self.without_propagation = configs.without_propagation
        self.without_cls = configs.without_cls
        self.without_contrastive = configs.without_contrastive
        self.without_future_rain = configs.without_future_rain
        self.use_real_rain = configs.use_real_rain
        adj_np = np.load('./dataset/adjacency_matrix.npy')
        self.register_buffer('adj_matrix', torch.tensor(adj_np, dtype=torch.float32))
        dist_np = np.load('./dataset/distance_matrix_km.npy')
        self.register_buffer('distance_matrix', torch.tensor(dist_np, dtype=torch.float32))
        flow_np = np.load('./dataset/flow_accum_weights.npy')
        self.register_buffer('flow_weights', torch.tensor(flow_np, dtype=torch.float32))
        slope_np = np.load('./dataset/slope_correlation_matrix.npy')
        self.register_buffer('slope_corr_matrix', torch.tensor(slope_np, dtype=torch.float32))
        self.feature_dim = None
        self.embed_dim = configs.d_model
        self.num_heads = configs.n_heads
        self.use_cls_plus_dyn = True
        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.beta1  = nn.Parameter(torch.tensor(1.0))
        self.alpha2 = nn.Parameter(torch.tensor(1.0))
        self.beta2  = nn.Parameter(torch.tensor(1.0))
        self.alpha3 = nn.Parameter(torch.tensor(1.0))
        self.beta3  = nn.Parameter(torch.tensor(1.0))
        self.static_d = 128
        self.static_projectors = nn.ModuleList([nn.Linear(1, self.static_d) for _ in range(5)])
        self.static_mlp = nn.Sequential(nn.Linear(self.static_d * 6, self.static_d), nn.ReLU())
        self.num_classes = 7
        self.class_embedding = nn.Embedding(self.num_classes, 128)
        self.flood_static_fusion = nn.Sequential(nn.Linear(configs.d_model + self.static_d, configs.d_model), nn.ReLU())
        self.enc_embedding_rain = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.enc_embedding_flood = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,configs.dropout)
        self.dec_embedding_rain = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding_flood = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.future_rain_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.future_rain_encoder = Encoder([
            EncoderLayer(AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads), configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation)
            for _ in range(configs.e_layers)
        ], norm_layer=torch.nn.LayerNorm(configs.d_model))
        if self.without_gate:
            self.rain_encoder_layers = nn.ModuleList([
                EncoderLayerWithStaticNogate(
                    AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, configs.dropout,
                    static_d=self.static_d
                ) for _ in range(configs.e_layers)
            ])
            self.flood_encoder_layers = nn.ModuleList([
                EncoderLayerWithStaticNogate(
                    AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, configs.dropout,
                    static_d=self.static_d
                ) for _ in range(configs.e_layers)
            ])
            self.rain_decoder = Decoder([
                DecoderLayer(
                    AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads),
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.d_layers)
            ], norm_layer=nn.LayerNorm(configs.d_model),
               projection=nn.Linear(configs.d_model, configs.c_out, bias=True))
            self.flood_decoder_layers = nn.ModuleList([
                DecoderLayerWithStaticNoGate(
                    AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads),
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, configs.dropout,
                    static_d=self.static_d
                ) for _ in range(configs.d_layers)
            ])
            self.future_rain_encoder_layers = nn.ModuleList([
                EncoderLayerWithStaticNogate(
                    AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, configs.dropout,
                    static_d=self.static_d
                ) for _ in range(configs.e_layers)
            ])
            self.flood_decoder_layers_future_rain = nn.ModuleList([
                DecoderLayerWithStaticWithFutureRain(
                    AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads),
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads),
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, configs.dropout,
                    static_d=self.static_d
                ) for _ in range(configs.d_layers)
            ])
        else:
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
            self.rain_decoder = Decoder([
                DecoderLayer(
                    AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads),
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.d_layers)
            ], norm_layer=nn.LayerNorm(configs.d_model),
               projection=nn.Linear(configs.d_model, configs.c_out, bias=True))
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
            self.future_rain_encoder_layers = nn.ModuleList([
                EncoderLayerWithStatic(
                    AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, configs.dropout,
                    static_d=self.static_d
                ) for _ in range(configs.e_layers)
            ])
            self.flood_decoder_layers_future_rain = nn.ModuleList([
                DecoderLayerWithStaticWithFutureRain(
                    AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads),
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads),
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, configs.dropout,
                    static_d=self.static_d
                ) for _ in range(configs.d_layers)
            ])
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.flood_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.flood_fc1 = nn.Linear(configs.d_model, 128)
        self.flood_fc2 = nn.Linear(128, configs.c_out)
        self.rain2flood = nn.Linear(7, 56)

    def forward(self, x_enc, future_rain, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        first = x_enc[:, :1].repeat(1, 7, 1, 1)
        rest = x_enc[:, 1:]
        x_enc = torch.cat((first, rest), dim=1)
        first = future_rain[:, :1].repeat(1, 7, 1)
        rest = future_rain[:, 1:]
        future_rain = torch.cat((first, rest), dim=1)
        first = x_dec[:, :1].repeat(1, 7, 1, 1)
        rest = x_dec[:, 1:]
        x_dec = torch.cat((first, rest), dim=1)
        x_enc = x_enc[..., [0, 2, 3, 4, 5, 6, 7]]
        x_dec = x_dec[..., [0]]
        adj = self.adj_matrix.to(x_enc.device).unsqueeze(0).unsqueeze(1)
        batch_size, node_num, seq_len, feature_dim = x_enc.shape
        dynamic_feature = x_enc[..., 0:1]
        static_continuous = x_enc[..., 1:6]
        static_class = x_enc[..., 6].long()
        if self.without_rain:
            self.without_future_rain=1
            enc_out_rain=torch.zeros(batch_size*7, seq_len, self.embed_dim, device=x_enc.device)
            flood_rain_effect=torch.zeros(batch_size,56, seq_len, 1, device=x_enc.device)
        else:
            rain_input = dynamic_feature[:, :7, :, :]
            if self.without_propagation:
                flood_rain_effect=torch.zeros(batch_size,56, seq_len, 1, device=x_enc.device)
            else:
                flood_rain_effect = self.compute_rainfall_influence(rain_input, adj,self.alpha1,self.beta1)
            rain_input=rain_input.reshape(batch_size * 7, seq_len, 1)
            x_mark_enc_rain=x_mark_enc.unsqueeze(1).repeat(1, 7, 1, 1)
            x_mark_enc_rain= x_mark_enc_rain.reshape(batch_size * 7, seq_len, 4)
            x_mark_enc_future_rain=x_mark_dec[:,-self.pred_len:,:].unsqueeze(1).repeat(1, 7, 1, 1)
            x_mark_enc_future_rain= x_mark_enc_future_rain.reshape(batch_size * 7, seq_len, 4)
            enc_out_rain = self.enc_embedding_rain(rain_input, x_mark_enc_rain)
            static_s_rain = torch.zeros_like(enc_out_rain[..., :self.static_d])
            xr = enc_out_rain
            for layer in self.rain_encoder_layers:
                xr, _ = layer(xr, static_s_rain)
            enc_out_rain = xr
        flood_input = dynamic_feature[:, 7:, :, :]+flood_rain_effect
        flood_input=flood_input.reshape(batch_size * 56, seq_len, 1)
        x_mark_enc_flood=x_mark_enc.unsqueeze(1).repeat(1, 56, 1, 1)
        x_mark_enc_flood= x_mark_enc_flood.reshape(batch_size * 56, seq_len,4)
        flood_embedding = self.enc_embedding_flood(flood_input, x_mark_enc_flood)
        if self.without_static:
            static_info_embedding = torch.zeros(batch_size, node_num, seq_len, self.static_d, device=x_enc.device)
        else:
            projected_static = []
            for i in range(5):
                feat = static_continuous[..., i:i+1]
                feat_proj = self.static_projectors[i](feat.reshape(-1,1))
                feat_proj = feat_proj.reshape(batch_size, node_num, seq_len, -1)
                projected_static.append(feat_proj)
            class_embeds = self.class_embedding(static_class).reshape(batch_size, node_num, seq_len, -1)
            static_embeds = torch.cat(projected_static + [class_embeds], dim=-1)
            static_info_embedding = self.static_mlp(static_embeds)
        if self.without_cls:
            flood_x=torch.zeros(batch_size*56, seq_len+1, self.embed_dim , device=x_enc.device)
            static_info_embedding = static_info_embedding[:, 7:, :, :]
        else:
            flood_embedding = flood_embedding.reshape(batch_size, 56, seq_len, -1)
            static_info_embedding = static_info_embedding[:, 7:, :, :]
            cls_token = self.flood_static_fusion(torch.cat([flood_embedding[:, :, :1, :], static_info_embedding[:, :, :1, :]], dim=-1))
            flood_token_seq = torch.cat([cls_token, flood_embedding], dim=2)
            flood_x = flood_token_seq.reshape(batch_size * 56, seq_len + 1, -1)
        static_s_full = torch.cat([torch.zeros(batch_size, 56, 1, self.static_d, device=x_enc.device), static_info_embedding], dim=2).reshape(batch_size * 56, seq_len + 1, self.static_d)
        xf = flood_x
        for layer in self.flood_encoder_layers:
            xf, _ = layer(xf, static_s_full)
        enc_out_flood = xf
        if self.without_propagation:
            enc_out_rain_effect=torch.zeros(batch_size,56, seq_len, 512, device=x_enc.device)
        else:
            enc_out_rain_effect=self.compute_rainfall_influence(enc_out_rain.reshape(batch_size,7,12,-1), adj,self.alpha2,self.beta2)
        enc_out_flood=enc_out_flood.reshape(batch_size,56,13,-1)
        enc_out_flood[:,:,1:,:]=enc_out_flood[:,:,1:,:]+enc_out_rain_effect
        enc_out_flood=enc_out_flood.reshape(batch_size*56,13,-1)
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
        rain_pred = dec_out_rain.view(batch_size, 7, self.pred_len)
        dec_f = dec_out_flood
        static_node_feat = static_info_embedding.mean(dim=2)
        dec_len = dec_f.size(1)
        static_s_dec = static_node_feat.unsqueeze(2).expand(batch_size, 56, dec_len, self.static_d)
        static_s_dec = static_s_dec.reshape(batch_size * 56, dec_len, self.static_d)
        if not self.without_future_rain:
            if self.use_real_rain:
                future_rain = future_rain.to(rain_pred.device).float()
                rain_pred=future_rain[:,:7,:]
            rain_pred_input = rain_pred.reshape(batch_size * 7, self.pred_len, 1)
            if x_mark_dec is None:
                future_rain_embed = self.future_rain_embedding(rain_pred_input, None)
            else:
                x_mark_future_rain = x_mark_dec[:, -self.pred_len:, :].unsqueeze(1).repeat(1, 7, 1, 1)
                x_mark_future_rain = x_mark_future_rain.reshape(batch_size * 7, self.pred_len, 4)
                future_rain_embed = self.future_rain_embedding(rain_pred_input, x_mark_future_rain)
            future_rain_encoded, _ = self.future_rain_encoder(future_rain_embed)
            future_rain_effect=self.rain2flood(future_rain_encoded.reshape(batch_size,7,seq_len,-1).permute(0, 2, 3, 1))
            future_rain_effect=future_rain_effect.permute(0, 3, 1, 2)
            future_rain_effect=future_rain_effect.reshape(batch_size*56,seq_len,-1)
            dec_f = dec_out_flood
            for layer in self.flood_decoder_layers_future_rain:
                dec_f = layer(dec_f, future_rain_effect, static_s_dec,enc_out_flood, self_mask=None, cross_mask=None)
        else:
            for layer in self.flood_decoder_layers:
                dec_f = layer(dec_f, enc_out_flood, static_s_dec, self_mask=None, cross_mask=None)
        mlp_input = dec_f[:, -self.pred_len:, :]
        mlp_out = F.relu(self.flood_fc1(mlp_input))
        dec_out_flood = self.flood_fc2(mlp_out)
        flood_pred = dec_out_flood.reshape(batch_size, 56, self.pred_len)
        pred_all = torch.cat([rain_pred, flood_pred], dim=1)
        if self.without_contrastive:
            reprs=torch.zeros(batch_size, 56, seq_len, self.embed_dim , device=x_enc.device)
        else:
            reprs=enc_out_flood[:,1:,:].reshape(batch_size,56,24,-1)
        return flood_pred, reprs

    def attention_fusion_concat(self, attn_outputs):
        fused = torch.cat(attn_outputs, dim=-1)
        return self.concat_projection(fused)

    def attention_fusion_gate(self, attn_outputs):
        gate = torch.softmax(self.gate_weights, dim=0)
        fused = sum(w * output for w, output in zip(gate, attn_outputs))
        return fused

    def compute_rainfall_influence(self, rain_encoded, adj_matrix, alpha=None, beta=None):
        B, _, T, D = rain_encoded.shape
        device = rain_encoded.device
        adj_matrix = adj_matrix[0, 0, :7, -56:]
        rain_to_flood = torch.einsum('sj,bsld->bjld', adj_matrix, rain_encoded)
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        flow_weights = self.flow_weights.to(device)[-56:, -56:]
        flood_rain_effect = []
        H_prev = torch.zeros(B, 56, D, device=device)
        for t in range(T):
            R_t = rain_to_flood[:, :, t, :]
            flow_acc = torch.einsum("ij,bjd->bid", flow_weights, H_prev)
            H_t = alpha * R_t + beta * flow_acc
            flood_rain_effect.append(H_t.unsqueeze(2))
            H_prev = H_t
        result=torch.cat(flood_rain_effect, dim=2) 
        return result