import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import LinearAnomalyAttention, AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention_layer = attention_layer
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x, queries, keys = self.attention_layer(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), queries, keys


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        queries_list = []
        keys_list = []
        for attn_layer in self.attn_layers:
            x, queries, keys = attn_layer(x)
            queries_list.append(queries)
            keys_list.append(keys)

        if self.norm is not None:
            x = self.norm(x)

        return x, queries_list, keys_list


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True, linear_attn=True, mapping_fun='ours'):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        dim_per_head = d_model//n_heads
        self.linear_attn = linear_attn
        # Encoder
        if self.linear_attn:
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            LinearAnomalyAttention(win_size, False, attention_dropout=dropout,
                                                   output_attention=output_attention, dim_per_head=dim_per_head,
                                                   mapping_fun=mapping_fun),
                            d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for _ in range(e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )
        else:
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            # LinearAnomalyAttention(win_size, False, attention_dropout=dropout,
                            #                        output_attention=output_attention, dim_per_head=dim_per_head),
                            AnomalyAttention(win_size, False, attention_dropout=dropout,
                                             output_attention=output_attention),
                            d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for _ in range(e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, queries_list, keys_list = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, queries_list, keys_list
        else:
            return enc_out  # [B, L, D]
