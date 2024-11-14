import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        tmp = torch.arange(win_size)
        self.distances = (tmp.unsqueeze(1)-tmp.unsqueeze(0)).abs().cuda()

    def forward(self, queries, keys, values):  # sigma, attn_mask
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        assert L == S, "L!=S"
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # if self.mask_flag:
        #     if attn_mask is None:
        #         attn_mask = TriangularCausalMask(B, L, device=queries.device)
        #     scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        # sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        # window_size = attn.shape[-1]
        # sigma = torch.sigmoid(sigma * 5) + 1e-5
        # sigma = torch.pow(3, sigma) - 1
        # sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        # prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        # prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("b h l s, b s h d -> b l h d", series, values)

        if self.output_attention:
            return V.contiguous(), series, None
        else:
            return V.contiguous(), None


class LinearAnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=False, scale=None, attention_dropout=0.0, output_attention=False,
                 dim_per_head=64, mapping_fun='ours'):
        super(LinearAnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        self.softmax = nn.Softmax(dim=-1)
        self.mapping_fun = mapping_fun

        self.delta1 = nn.Parameter(torch.tensor(1.0))
        # self.delta2 = nn.Parameter(torch.tensor(1.0))
        # self.scale = nn.Parameter(torch.zeros(size=(1, 1, 1, dim_per_head)))

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        assert L == S, "L!=S"

        if self.mapping_fun == 'ours':
            queries[queries < 0] = -100
            keys[keys < 0] = -100
            queries = self.softmax(queries / nn.Softplus()(self.delta1))
            keys = self.softmax(keys / nn.Softplus()(self.delta1))
        elif self.mapping_fun == 'softmax_q_k':
            # softmax2
            queries = self.softmax(queries)
            softmax2 = nn.Softmax(dim=1)
            keys = softmax2(keys)
        elif self.mapping_fun == 'x_3':
            # x**3 and relu
            queries = nn.ReLU()(queries)
            keys = nn.ReLU()(keys)
            # x**3
            q_norm = queries.norm(dim=-1, keepdim=True)
            k_norm = keys.norm(dim=-1, keepdim=True)
            queries = queries**3
            keys = keys**3
            queries = queries / (queries.norm(dim=-1, keepdim=True)+1e-6) * q_norm.clone()
            keys = keys / (keys.norm(dim=-1, keepdim=True) + 1e-6) * k_norm.clone()
        elif self.mapping_fun == 'relu':
            queries = nn.ReLU()(queries)
            keys = nn.ReLU()(keys)
        elif self.mapping_fun == 'elu_plus_1':
            # elu+1
            queries = F.elu(queries) + 1
            keys = F.elu(keys) + 1

        kv = torch.einsum("b e h l, b l h f -> b h e f", keys.transpose(1, 3), values)

        z = 1 / (torch.einsum("b l h e, b h e -> b l h", queries, keys.sum(dim=1)) + 1e-6)
        V = torch.einsum("b l h e, b h e e, b l h -> b l h e", queries, kv, z)

        if self.output_attention:
            return V.contiguous(), queries, keys
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, queries, keys = self.inner_attention(
            queries,
            keys,
            values
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), queries, keys
