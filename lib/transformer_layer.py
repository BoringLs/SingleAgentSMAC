import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention


class ScaleDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1, bias=-1e9):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.bias = bias

    def forward(self, q, k, v, mask=None):

        scores = torch.matmul(q, k.transpose(-1, -2)) / self.temperature
        # 本项目中mask表示存活的单位，所以mask=0的地方要被mask掉
        if mask is not None:
            scores = scores.masked_fill(mask == 0, self.bias)
        atten = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(atten, v)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)

        self.attention = ScaleDotProductAttention(temperature=d_k**0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        batch_size = q.size(0)
        seq_len = q.size(1)

        q = (
            self.w_qs(q)
            .view(batch_size, seq_len, self.n_head, self.d_k)
            .transpose(1, 2)
        )
        k = (
            self.w_ks(k)
            .view(batch_size, seq_len, self.n_head, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.w_vs(v)
            .view(batch_size, seq_len, self.n_head, self.d_v)
            .transpose(1, 2)
        )

        if mask is not None:
            mask = mask.unsqueeze(1)

        output = self.attention(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.fc(output)
        output = self.dropout(output)

        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_hid)
        self.w_2 = nn.Linear(d_hid, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = F.relu(self.w_1(x))
        output = self.w_2(output)
        output = self.dropout(output)

        return output
