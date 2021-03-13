"""
    비트코인 투자를 위한 지도학습 투자 모델

    @author: Younghyun Kim
    Created on 2021.01.10
"""
import math
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Uniform


class SuperInvestor(nn.Module):
    """
        Super Investor
    """
    def __init__(self, input_dim=6, seq_length=1440,
                 emb_dim=4, hidden_dim=16, nheads=2, dropout=0.1):
        super(SuperInvestor, self).__init__()

        self.input_dim = input_dim
        self.seq_length = seq_length
        self.emb_dim = emb_dim
        self.nheads = nheads
        self.dropout = dropout

        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

        self.emb_net = nn.Linear(input_dim, emb_dim)
        self.attn = CrossSectionalTransformer(emb_dim, nheads, dropout)
        self.pos_enc = PositionalEncoding(emb_dim, dropout)

        tshape = int(seq_length * emb_dim)
        self.act_net1 = nn.Linear(tshape, int(tshape / 2))
        self.act_net2 = nn.Linear(int(tshape / 2), hidden_dim)
        self.act_net3 = nn.Linear(hidden_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        embs = self.gelu(self.emb_net(x))
        embs = self.pos_enc(embs)

        attn = self.attn(embs)
        attn = attn.view(x.shape[0], -1)

        acts = self.gelu(self.act_net1(attn))
        acts = self.gelu(self.act_net2(acts))
        acts = -self.relu(-self.act_net3(acts))

        return acts


class CrossSectionalTransformer(nn.Module):
    """
        Cross Sectional Transformer
    """
    def __init__(self, asset_dim=4, nheads=2, dropout=0.1):
        super(CrossSectionalTransformer, self).__init__()
        self.asset_dim = asset_dim
        self.nheads = nheads
        self.dim_feedforward = int(asset_dim * 2)
        self.dropout = dropout

        self.encoder =\
            nn.TransformerEncoderLayer(asset_dim, nheads,
                                       dim_feedforward=self.dim_feedforward,
                                       dropout=dropout)

    def forward(self, assets):
        assets = self.encoder(assets)

        return assets


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000, peps=0.01):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.peps = peps

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) * peps
        self.register_buffer('pe', pe)

    def forward(self, x):
        x_pos = x + Variable(self.pe[:, :x.size(1)],
                             requires_grad=False)
        return self.dropout(x_pos)
