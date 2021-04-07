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

CONFIG = {
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'trajectory_size': 2049,
            'lr': 1e-4,
            'ppo_eps': 0.2,
            'ppo_epoches': 10,
            'ppo_batch_size': 64,
            'test_iters': 1000
         }


class FFPTPPOTrainer:
    """
        PPO Trainer for FFP Trader
    """
    def __init__(self, ffp_env, config=CONFIG):
        " Initiation "
        self.env = ffp_env

        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.trajectory_size = config['trajectory_size']
        self.lr = config['lr']

        self.ppo_eps = config['ppo_eps']
        self.ppo_epoches = config['ppo_epoches']
        self.ppo_batch_size = config['ppo_batch_size']
        self.test_iters = config['test_iters']
