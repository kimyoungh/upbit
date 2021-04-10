"""
    PPO Trainer for FFPTrader

    Created on 2021.04.07
    @author: Younghyun Kim
"""
import os
import math
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from environ import Environ, env_config
from ffp_trader import FFPTrader

TRAINER_CONFIG = {
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'trajectory_size': 128,
            'lr': 1e-4,
            'ppo_eps': 0.2,  # clipping value for (new / old)
            'ppo_epoches': 10,
            'ppo_batch_size': 64,
            'test_iters': 1000,
            'device': 'cuda:0',
         }


class FFPTPPOTrainer:
    """
        PPO Trainer for FFP Trader
    """
    def __init__(self, trader, ffp_env, config=TRAINER_CONFIG):
        " Initiation "
        self.trader = trader  # FFPTrader
        self.env = ffp_env  # Environ

        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.trajectory_size = config['trajectory_size']
        self.lr = config['lr']

        self.ppo_eps = config['ppo_eps']
        self.ppo_epoches = config['ppo_epoches']
        self.ppo_batch_size = config['ppo_batch_size']
        self.test_iters = config['test_iters']
        self.device = config['device']

        self.trader = self.trader.to(self.device)

        self.opt = optim.Adam(self.trader.parameters(),
                              lr=self.lr)

    @staticmethod
    def calc_logprob(mu_v, logstd_v, actions_v, cmin=1e-3):
        p1 = - ((mu_v - actions_v) ** 2) /\
                (2 * torch.exp(logstd_v).clip(min=cmin))
        p2 = - torch.log(
            torch.sqrt(2 * math.pi * torch.exp(logstd_v)))

        return p1 + p2

    def calc_adv_ref(self, trajectory, values_v, device='cpu'):
        """
            Calculation for advantage and 1-step ref value

            Args:
                trajectory: trajectory list
                value_v: critic values tensor
            Return:
                tuple with advantage numpy array and
                reference values
        """
        values = values_v.squeeze().data.cpu().numpy()

        # generalized advantage estimator:
        #  smoothed version of the advantage
        last_gae = 0.0
        result_adv, result_ref = [], []

        for val, next_val, exp in\
            zip(reversed(values[:-1]), reversed(values[1:]),
                reversed(trajectory[:-1])):
            if exp[-1]:  # done
                delta = exp[2] - val  # exp[2]: reward
                last_gae = delta
            else:
                delta = exp[2] + self.gamma * next_val - val
                last_gae =\
                    delta + self.gamma * self.gae_lambda * last_gae
            result_adv.append(last_gae)
            result_ref.append(last_gae + val)

        adv_v = torch.FloatTensor(
                list(reversed(result_adv))).to(device)
        ref_v = torch.FloatTensor(
                list(reversed(result_ref))).to(device)

        return adv_v, ref_v

    def train(self, episodes=20000):
        """
            Train PPO

            Args:
                episodes: # of episodes
        """
