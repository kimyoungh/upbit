"""
    Super Investor Trainer Module

    @author: Younghyun Kim
    Created on 2021.01.10
"""
import os
import pickle
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.nn.utils as nn_utils

from tensorboardX import SummaryWriter

from super_allocator import SuperInvestor

COLUMNS = ['opening_price', 'high_price', 'low_price', 'trade_price',
           'candle_acc_trade_price', 'candle_acc_trade_volume']
EPS = 1e-6
COST = 0.004


class SuperTrainer:
    " Super Trainer "
    def __init__(self, model, price_data, seq_length=1440, target_length=30,
                 lr=0.00001, clip_grad=0.5, cost=0.004,
                 logdir='./logdir/super/',
                 model_path='./models/super_trader.pt', device='cuda:0'):
        self.model = model

        if torch.cuda.is_available() and device[:4] == 'cuda':
            self.device = device
        else:
            self.device = 'cpu'
        self.model = self.model.to(self.device)
        self.model.eval()

        self.seq_length = seq_length
        self.target_length = target_length
        self.lr = lr
        self.clip_grad = clip_grad

        self.cost = cost
        self.price_data = price_data[COLUMNS]

        self.price_data.insert(6, 'high_low',
                               price_data['high_price'] - price_data['low_price'],
                               True)
        self.price_data.insert(7, 'high_trade',
                               price_data['high_price'] - price_data['trade_price'],
                               True)
        self.price_data.insert(8, 'trade_low',
                               price_data['trade_price'] - price_data['low_price'],
                               True)
        self.price_data.insert(9, 'open_close',
                               price_data['opening_price'] - price_data['trade_price'],
                               True)
        self.price_data.insert(10, 'open_high',
                               price_data['opening_price'] - price_data['high_price'],
                               True)
        self.price_data.insert(11, 'open_low',
                               price_data['opening_price'] - price_data['low_price'],
                               True)
        self.price = self.price_data.copy().values

        self.returns =\
            np.log(self.price_data['trade_price']).diff(target_length)
        self.target = (self.returns > 0.).astype(int).values

        self.pos_idx = np.argwhere(self.target == 1).ravel() - (seq_length - 1) - target_length
        self.neg_idx = np.argwhere(self.target == 0).ravel() - (seq_length - 1) - target_length

        self.pos_idx = self.pos_idx[self.pos_idx >= 0]
        self.neg_idx = self.neg_idx[self.neg_idx >= 0]

        self.eps = EPS

        self.logdir = logdir
        self.model_path = model_path

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, eps=1e-3)
        self.loss_fn = torch.nn.MSELoss()

    def _calc_minmax(self, data, eps=1e-6):
        """
            Calculate Minmax Scailing

            Args:
                data: np.array, price time series
        """
        dmin = data.min(axis=0)
        dmax = data.max(axis=0)

        normalized = (data - dmin) / (dmax - dmin + eps)

        return normalized

    def train(self, iteration=500000, batch_size=64,
              scheduling=True, sche_term=10000., gamma=0.95,
              val_prob=0.2, val_batch=32, seed=0):
        """
            train method
            batch_size: 무조건 짝수!
        """
        writer = SummaryWriter(logdir=self.logdir)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_results = []

        val_N = round(self.price.shape[0] * val_prob)
        val_pos_idx = np.random.choice(self.pos_idx, int(val_N / 2),
                                       replace=False)
        val_neg_idx = np.random.choice(self.neg_idx, int(val_N / 2),
                                       replace=False)
        train_pos_idx = np.setdiff1d(self.pos_idx, val_pos_idx)
        train_neg_idx = np.setdiff1d(self.neg_idx, val_neg_idx)

        if scheduling:
            scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                  sche_term,
                                                  gamma=gamma)

        for i in range(iteration):
            self.model.train()
            train_pos_is = np.random.choice(train_pos_idx, int(batch_size / 2),
                                            replace=False)
            train_neg_is = np.random.choice(train_neg_idx, int(batch_size / 2),
                                            replace=False)
            train_is = np.stack((train_pos_is, train_neg_is), axis=0).ravel()
            np.random.shuffle(train_is)
            trainset = []
            target = []

            self.optimizer.zero_grad()

            for j in train_is:
                data = self.price[j:j+self.seq_length]
                pdata = self._calc_minmax(data)
                trainset.append(pdata)
            trainset = torch.FloatTensor(trainset).to(self.device)
            target = self.target[train_is + (self.seq_length - 1) +
                                 self.target_length]
            target = torch.FloatTensor(target).to(self.device)
            target = target.view(-1, 1)

            pred = torch.exp(self.model(trainset))

            loss = self.loss_fn(pred, target) * 100.
            loss.backward()

            self.optimizer.step()

            if scheduling:
                scheduler.step()

            # Validation
            if val_N > 0:
                self.model.eval()
                val_pos_is = np.random.choice(val_pos_idx, val_batch,
                                              replace=False)
                val_neg_is = np.random.choice(val_neg_idx, val_batch,
                                              replace=False)
                val_is = np.stack((val_pos_is, val_neg_is), axis=0).ravel()
                np.random.shuffle(val_is)
                valset = []
                val_target = []

                for j in val_is:
                    data = self.price[j:j+self.seq_length]
                    pdata = self._calc_minmax(data)
                    valset.append(pdata)
                valset = torch.FloatTensor(valset).to(self.device)
                val_target = self.target[val_is + (self.seq_length - 1) +
                                         self.target_length]
                val_target = torch.FloatTensor(val_target).to(self.device)
                val_target = val_target.view(-1, 1)

                with torch.no_grad():
                    val_pred = torch.exp(self.model(valset))

                    val_loss = self.loss_fn(val_pred, val_target) * 100.

            writer.add_scalars('loss', {'train': loss.item(),
                                        'val': val_loss.item()}, i)
            torch.save(self.model.state_dict(), self.model_path)
            print(i, "Loss: " + str(loss.item()),
                  "val_loss: " + str(val_loss.item()))
            train_results.append([loss.item(), val_loss.item()])
        train_results = pd.DataFrame(train_results, columns=['Train', 'Validation'])
        with open('train_results.pkl', 'wb') as f:
            pickle.dump(train_results, f)


if __name__ == "__main__":
    with open('bitcoin_price_recent.pkl', 'rb') as f:
        price = pickle.load(f)

    iters = int(input("Type Iteration #: "))
    model = SuperInvestor(12, 60, 8, 32, 2, 0.1)

    trainer = SuperTrainer(model, price, 60, 5, 0.001, clip_grad=0.5,
                           logdir='./logdir/super/',
                           model_path='./models/super_trader.pt',
                           device='cuda:0')

    trainer.train(iteration=iters, batch_size=64,
                  scheduling=True, sche_term=10000., gamma=0.95,
                  val_prob=0.2, val_batch=32, seed=0)
