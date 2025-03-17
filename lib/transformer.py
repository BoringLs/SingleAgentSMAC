from torch import nn
from lib.transformer_layer import *


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        d_inner=1024,
        n_head=8,
        d_k=64,
        d_v=64,
        n_layers=5,
        droput=0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, d_inner, n_head, d_k, d_v, droput)
                for _ in range(n_layers)
            ]
        )

    def forward(self, input, mask=None):
        for layer in self.layers:
            input = layer(input, mask)
        return input


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout):
        super().__init__()

        self.self_atten = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_inner, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, input, mask=None):
        residual = input
        output = self.self_atten(input, input, input, mask)
        output = self.dropout1(output)
        output = self.layernorm1(output + residual)

        residual = output
        output = self.feed_forward(output)
        output = self.dropout2(output)
        output = self.layernorm2(output + residual)

        return output
