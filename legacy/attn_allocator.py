"""
    비트코인 투자를 위한 포트폴리오 모델 + Transformer
    현금 + 비트코인 배분 모델

    @author: Younghyun Kim
    Created on 2021.01.01
"""
import math
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Uniform


class BitInvestor(nn.Module):
    """
        Bit Investor
    """
    def __init__(self, prsi_dim=32, trsi_dim=8,
                 pm_dim=8, vol_dim=8, tvol_dim=8,
                 hidden_dim=4, nheads=2, dropout=0.1):
        super(BitInvestor, self).__init__()

        self.prsi_dim = prsi_dim
        prsi_dim2 = int(prsi_dim / 2)
        self.trsi_dim = trsi_dim
        self.pm_dim = pm_dim
        self.vol_dim = vol_dim
        self.tvol_dim = tvol_dim
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.dropout = dropout

        tshape = 5 * hidden_dim
        self.spike = Uniform(0., 1.)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

        self.prsi_net1 = nn.Linear(prsi_dim, prsi_dim2)
        self.prsi_net2 = nn.Linear(prsi_dim2, hidden_dim)
        self.trsi_net = nn.Linear(trsi_dim, hidden_dim)
        self.pm_net = nn.Linear(pm_dim, hidden_dim)
        self.vol_net = nn.Linear(vol_dim, hidden_dim)
        self.tvol_net = nn.Linear(tvol_dim, hidden_dim)

        self.attn = CrossSectionalTransformer(hidden_dim, nheads, dropout)
        self.wnet = nn.Linear(tshape, 2)
        self.qnet = nn.Linear(tshape, 2)
        self.renet = nn.Linear(tshape, 1)

    def get_rebal_msk(self, log_rebal, train=False,
                      init=False, num=100.):
        if train:
            spike = self.spike.rsample(log_rebal.shape)
            spike = spike.to(log_rebal.device)
            if init:
                rebal_msk = self.sigmoid(num * (torch.exp(log_rebal) + 0.5))
            else:
                rebal_msk = self.sigmoid(num * (spike.detach() \
                        - (1. - torch.exp(log_rebal))))
        else:
            if init:
                rebal_msk = torch.tensor(1.).to(log_rebal.device)
            else:
                rebal_msk = torch.round(torch.exp(log_rebal))

        return rebal_msk

    def sel_rebalancing(self, bemb, bweights_prev, train=False,
                        init=False, num=100.):
        port_now = bweights_prev[0, 0].item() * bemb
        log_rebal = -self.relu(-self.renet(port_now))
        rebal_msk = self.get_rebal_msk(log_rebal, train=train,
                                       init=init, num=num)

        return rebal_msk.view(-1)

    def forward(self, prsi, trsi, pm, vol, tvol, weights_prev,
                init=False, train=False, num=100.):
        if train:
            self.train()

            prsi = self.gelu(self.prsi_net1(prsi))
            prsi = self.gelu(self.prsi_net2(prsi)).unsqueeze(1)
            trsi = self.gelu(self.trsi_net(trsi)).unsqueeze(1)
            pm = self.gelu(self.pm_net(pm)).unsqueeze(1)
            vol = self.gelu(self.vol_net(vol)).unsqueeze(1)
            tvol = self.gelu(self.tvol_net(tvol)).unsqueeze(1)

            bits = torch.cat((prsi, trsi, pm, vol, tvol), dim=1)
            bits = self.attn(bits)
            bits = bits.view(bits.shape[0], -1)

            rebal_msk = self.sel_rebalancing(bits, weights_prev,
                                             train=train, init=init, num=num)
            wscores = self.sigmoid(self.wnet(bits)) * rebal_msk
            q_values = self.qnet(bits)

            if rebal_msk.item() > 0.5:
                weights = wscores / wscores.sum()
            else:
                weights = torch.zeros_like(wscores).detach()
            nweights = weights_prev * (1. - rebal_msk)
        else:
            self.eval()
            with torch.no_grad():
                prsi = self.gelu(self.prsi_net1(prsi))
                prsi = self.gelu(self.prsi_net2(prsi)).unsqueeze(1)
                trsi = self.gelu(self.trsi_net(trsi)).unsqueeze(1)
                pm = self.gelu(self.pm_net(pm)).unsqueeze(1)
                vol = self.gelu(self.vol_net(vol)).unsqueeze(1)
                tvol = self.gelu(self.tvol_net(tvol)).unsqueeze(1)

                bits = torch.cat((prsi, trsi, pm, vol, tvol), dim=1)
                bits = self.attn(bits)
                bits = bits.view(bits.shape[0], -1)

                rebal_msk = self.sel_rebalancing(bits, weights_prev,
                                                 train=train, init=init, num=num)
                wscores = self.sigmoid(self.wnet(bits)) * rebal_msk
                q_values = self.qnet(bits)

            if rebal_msk.item() > 0.5:
                weights = wscores / wscores.sum()
            else:
                weights = torch.zeros_like(wscores).detach()
            nweights = weights_prev * (1. - rebal_msk)

            weights = weights.detach()
            wscores = wscores.detach()
            q_values = q_values.detach()
            rebal_msk = rebal_msk.detach()

        rweights = torch.cat((nweights, weights), dim=1)

        return rweights, wscores, q_values, rebal_msk


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
