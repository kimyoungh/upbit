"""
    비트코인 투자를 위한 포트폴리오 모델
    현금 + 비트코인 배분 모델

    @author: Younghyun Kim
    Created on 2020.11.01
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
    def __init__(self, in_channels=6, mlength=60, moutdim=4,
                 hlength=24, houtdim=4,
                 dlength=250, doutdim=4):
        super(BitInvestor, self).__init__()
        self.in_channels = in_channels

        self.mlength = mlength
        self.moutdim = moutdim

        self.hlength = hlength
        self.houtdim = houtdim

        self.dlength = dlength
        self.doutdim = doutdim

        tshape = moutdim + houtdim + doutdim
        self.spike = Uniform(0., 1.)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.mcnn = SequenceCNN(in_channels, mlength, moutdim)
        self.hcnn = SequenceCNN(in_channels, hlength, houtdim)
        self.dcnn = SequenceCNN(in_channels, dlength, doutdim)

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
                rebal_msk = torch.tensor(1.).to(log_rebal_device)
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

    def forward(self, minutes, hours, days, weights_prev,
                init=False, train=False, num=100.):
        if train:
            self.train()

            minutes = self.mcnn(minutes)
            hours = self.hcnn(hours)
            days = self.dcnn(days)

            bits = torch.cat((minutes, hours, days), dim=1)

            rebal_msk = self.sel_rebalancing(bits, weights_prev,
                                             train=train, init=init, num=num)
            wscores = self.sigmoid(self.wnet(bits)) * rebal_msk
            q_values = self.qnet(bits)

            if rebal_msk.item() > 0.:
                weights = wscores / wscores.sum()
            else:
                weights = torch.zeros_like(wscores).detach()
        else:
            self.eval()
            with torch.no_grad():
                minutes = self.mcnn(minutes)
                hours = self.hcnn(hours)
                days = self.dcnn(days)

                bits = torch.cat((minutes, hours, days), dim=1)

                rebal_msk = self.sel_rebalancing(bits, weights_prev,
                                                 train=train, init=init, num=num)
                wscores = self.sigmoid(self.wnet(bits)) * rebal_msk
                q_values = self.qnet(bits)

            if rebal_msk.item() > 0.:
                weights = wscores / wscores.sum()
            else:
                weights = torch.zeros_like(wscores)

                weights = weights.detach()
                wscores = wscores.detach()
                q_values = q_values.detach()
                rebal_msk = rebal_msk.detach()

        rweights = torch.cat((rebal_msk.unsqueeze(0), weights), dim=1)

        return rweights, wscores, q_values

class SequenceCNN(nn.Module):
    """
        Sequence CNN
    """
    def __init__(self, in_channels=7, length=60, output_dim=4):
        super(SequenceCNN, self).__init__()
        self.in_channels = in_channels
        self.length = length
        self.output_dim = output_dim

        self.conv1 = nn.Conv1d(in_channels, 8, 4, 2)
        self.conv2 = nn.Conv1d(8, 16, 4, 2)
        self.conv3 = nn.Conv1d(16, 32, 4, 2)

        self.act = nn.LeakyReLU(0.2)

        self.outdim = self._calc_outshape()

        self.fc1 = nn.Linear(self.outdim * 32, output_dim * 2)
        self.fc2 = nn.Linear(output_dim * 2, output_dim)

    def _calc_outshape(self):
        d1 = (self.length - (4 - 1) - 1) / 2 + 1
        d2 = (d1 - (4 - 1) - 1) / 2 + 1
        d3 = (d2 - (4 - 1) - 1) / 2 + 1

        return int(d3)

    def forward(self, x):
        hidden = self.conv1(x)
        hidden = self.act(hidden)
        hidden = self.conv2(hidden)
        hidden = self.act(hidden)
        hidden = self.conv3(hidden)
        hidden = self.act(hidden)

        hidden = hidden.view(-1, self.outdim * 32)
        out = self.fc1(hidden)
        out = self.act(out)
        out = self.fc2(out)

        return out
