"""
    Bitcoin 매매 학습을 위한 환경

    Created on 2020.11.01
    @author: Younghyun Kim
"""
import numpy as np
import pandas as pd

import torch


class Environment:
    """
        Bitcoin 매매 학습을 위한 게임 환경 조성 클래스
    """
    def __init__(self, price_data, tlength=1440,
                 buying_fee=0.004, selling_fee=0.004):
        """
            Initialization
            Args:
                    price_data: dataframe. 비트코인 시계열 데이터
                    tlength: 투자 학습 진행 시간(default: 1D(1440min))
        """
        cols_n = [2, 3, 4, 5, 6, 8, 9]
        cols = price_data.columns[cols_n]

        self.price_data = price_data[cols].set_index('candle_date_time_kst')
        self.dlength = price_data.shape[0]
        self.returns = self.price_data['trade_price'].pct_change()
        self.retmax = self.returns.cummax()

        self.min_beg = 250 * 24 * 60 - 1
        self.beg_pos = self.min_beg

        self.tlength = tlength

        self.cur_pos = self.beg_pos
        self.end_pos = None

        self.buying_fee = buying_fee
        self.selling_fee = selling_fee

    def reset(self, beg_pos=None):
        """
            reset

            Args:
                    beg_pos: int. 투자 시작 시점
            Return:
                    obs: torch.FloatTensor
                        [minutes, hours, days]
        """
        if beg_pos is not None:
            self.beg_pos = max(self.min_beg, beg_pos)

        self.cur_pos = self.beg_pos
        self.end_pos = min(self.cur_pos + self.tlength, self.dlength - 1)

        obs = self._calc_obs()

        done = False

        return obs, done

    def step(self, action, weights_prev, init=False):
        """
            step.
            Args:
                action: torch.tensor. [rebalancing, bit, cash]
                weights_prev: latest weights
            Return:
                next_obs, reward, return, latest_w_prev, done
                    * latest_w_prev: 리밸런싱을 하면 None
                                     리밸런싱을 안하면 이전 비중 * return
        """
        if action.device != 'cpu':
            acts = action.detach().cpu().numpy()
            w_prev = weights_prev.detach().cpu().numpy()
        else:
            acts = action.detach().numpy()
            w_prev = weights_prev.detach().numpy()

        # 리밸런싱 안할 경우에 최신 weight로 업데이트하기 위한 수익률
        wret = self.returns.iloc[self.cur_pos]

        latest_w_prev = w_prev * [1. + wret, 1.]

        if not init and w_prev.sum() > 0.:
            latest_w_prev /= latest_w_prev.sum()

        self.cur_pos += 1
        ret = self.returns.iloc[self.cur_pos]

        if acts[0, -2:].sum() > 0.5:
            weights = acts[0, -2:]
            w_diff = weights - latest_w_prev

            if w_diff[0] > 0.:
                fee = self.buying_fee * w_diff[0]
            elif w_diff[0] <= 0.:
                fee = self.selling_fee * w_diff[0]

            ret -= fee

            latest_w_prev = None
        else:
            weights = acts[0, :2]

        if ret > 0 and weights[0] >= weights[1]:
            reward = 1
        elif ret > 0 and weights[0] < weights[1]:
            reward = -1
        elif ret <= 0 and weights[0] >= weights[1]:
            reward = -1
        elif ret <= 0 and weights[0] < weights[1]:
            reward = 1

        next_obs = self._calc_obs()

        if self.cur_pos == self.end_pos:
            done = True
        else:
            done = False

        return next_obs, reward, ret, latest_w_prev, done

    def _calc_obs(self, eps=1e-6):
        " calculate observation "

        data = self.price_data

        # minutes
        minutes = data.iloc[self.cur_pos - 60 + 1:self.cur_pos + 1].copy()
        minutes = minutes / (minutes.max() + eps)
        minutes = torch.FloatTensor(minutes.values.astype(float))
        minutes = minutes.unsqueeze(0).transpose(1, 2)

        # hours
        hours = data.iloc[self.cur_pos - 60 * 24 + 1:self.cur_pos + 1].copy()
        vols = hours[hours.columns[-2:]]
        vols = vols.rolling(60).sum()
        hours.loc[:, hours.columns[-2:]] = vols
        hours = hours.iloc[60 * np.arange(24) + 59]
        hours = hours / (hours.max() + eps)
        hours = torch.FloatTensor(hours.values.astype(float))
        hours = hours.unsqueeze(0).transpose(1, 2)

        # days
        days = data.iloc[self.cur_pos - 60 * 24 * 250 + 1:self.cur_pos + 1].copy()
        vols = days[days.columns[-2:]]
        vols = vols.rolling(60 * 24).sum()
        days.loc[:, days.columns[-2:]] = vols
        days = days.iloc[60 * 24 * np.arange(250) + (60 * 24 - 1)]
        days = days / (days.max() + eps)
        days = torch.FloatTensor(days.values.astype(float))
        days = days.unsqueeze(0).transpose(1, 2)

        return [minutes, hours, days]
