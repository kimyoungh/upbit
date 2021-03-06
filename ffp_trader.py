"""
    Transformer와 RL을 활용한 비트코인 트레이딩 모델

    Created on 2021.04.06
    @author Younghyun Kim
"""
import math
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Uniform


class FFPTrader(nn.Module):
    """
        Trader for Financial-Freedom Project
    """
    def __init__(self, order_dim=30, channel=2,
                 emb_dim=4, total_bidask_dim=2,
                 price_dim=1, action_dim=3,
                 nheads=2, dropout=0.1):
        super(FFPTrader, self).__init__()

        self.order_dim = order_dim
        self.channel = channel
        self.emb_dim = emb_dim
        self.total_bidask_dim = total_bidask_dim
        self.price_dim = price_dim
        self.action_dim = action_dim
        self.nheads = nheads
        self.dropout = dropout

        self.gelu = nn.GELU()
        self.softplus = nn.Softplus()

        self.order_conv = nn.Sequential(
                            nn.Conv1d(channel, 4, 4, 1),
                            nn.GELU(),
                            nn.Conv1d(4, 8, 4, 1),
                            nn.GELU(),
                            nn.Conv1d(8, 16, 2, 1),
                            nn.GELU())

        self.bidask_fc = nn.Sequential(
                            nn.Linear(total_bidask_dim,
                                      total_bidask_dim),
                            nn.GELU(),
                            nn.Linear(total_bidask_dim, 1),
                            nn.GELU())

        self.price_fc = nn.Sequential(
                            nn.Linear(price_dim, price_dim),
                            nn.GELU(),
                            nn.Linear(price_dim, 1),
                            nn.GELU())

        with torch.no_grad():
            x = torch.randn(1, channel, order_dim)
            x = self.order_conv(x)
            x_dim = self.num_flat_features(x)

        self.x_dim = x_dim

        self.emb_net = nn.Linear(x_dim + 1 + 1 + 1, emb_dim)
        self.attn = Transformer(emb_dim, nheads, dropout)
        self.pos_enc = PositionalEncoding(emb_dim, dropout)

        self.policy_mu = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, 4),
                nn.GELU(),
                nn.Linear(4, action_dim)
        )

        self.policy_logstd = nn.Parameter(torch.zeros(action_dim))

        self.value = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, 4),
                nn.GELU(),
                nn.Linear(4, 1)
        )

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, orderbook, total_bid_ask,
                trade_price, possession_series):
        orders =\
            torch.zeros(orderbook.shape[0],
                        orderbook.shape[1], self.x_dim).to(orderbook.device)

        for i in range(orders.shape[0]):
            for j in range(orders.shape[1]):
                orders[i, j] = self.order_conv(orderbook[i, j].unsqueeze(0)).\
                        view(-1, 1, self.x_dim)
        bidask = self.bidask_fc(total_bid_ask)
        price =\
            self.price_fc(trade_price.view(orders.shape[0],
                                           -1, 1))

        embs = torch.cat((orders, bidask,
                          price,
                          possession_series.view(orders.shape[0],
                                                 -1, 1)), dim=-1)
        embs = self.gelu(self.emb_net(embs))
        embs = self.pos_enc(embs)

        attn = self.attn(embs)
        attn = attn.mean(1)

        policy_mu = self.policy_mu(attn)

        return policy_mu, self.value(attn)


class Transformer(nn.Module):
    """
        Transformer
    """
    def __init__(self, emb_dim=4, nheads=2, dropout=0.1):
        super(Transformer, self).__init__()
        self.emb_dim = emb_dim
        self.nheads = nheads
        self.dim_feedforward = int(emb_dim * 2)
        self.dropout = dropout

        self.encoder =\
            nn.TransformerEncoderLayer(emb_dim, nheads,
                                       dim_feedforward=self.dim_feedforward,
                                       dropout=dropout)

    def forward(self, embs):
        embs = self.encoder(embs)

        return embs


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
