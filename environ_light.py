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
                 buying_fee=0.004, selling_fee=0.004,
                 eps=1e-6):
        """
            Initialization
            Args:
                    price_data: dataframe. 비트코인 시계열 데이터
                    tlength: 투자 학습 진행 시간(default: 1D(1440min))
        """
        cols_n = [2, 3, 4, 5, 6, 8]
        cols = price_data.columns[cols_n]

        price_data = price_data[cols].set_index('candle_date_time_kst')

        self.price = price_data[price_data.columns[:-1]]
        self.turnover = price_data[['candle_acc_trade_price']]

        self.pdiff = self.price.diff(1)
        self.tdiff = self.turnover.diff(1)

        self.returns = np.log(price_data['trade_price']).diff(1)
        self.treturns = np.log(price_data['candle_acc_trade_price'] +
                               eps).diff(1)

        self.dlength = price_data.shape[0]

        # Price RSI 계산
        prsi_5 = self._calc_rsi(self.pdiff, 5)
        prsi_10 = self._calc_rsi(self.pdiff, 10)
        prsi_30 = self._calc_rsi(self.pdiff, 30)
        prsi_60 = self._calc_rsi(self.pdiff, 60)
        prsi_120 = self._calc_rsi(self.pdiff, 120)
        prsi_360 = self._calc_rsi(self.pdiff, 360)
        prsi_720 = self._calc_rsi(self.pdiff, 720)
        prsi_1440 = self._calc_rsi(self.pdiff, 1440)

        self.prsi = pd.concat((prsi_5, prsi_10, prsi_30,
                               prsi_60, prsi_120, prsi_360,
                               prsi_720, prsi_1440), axis=1)

        # Turnover RSI
        trsi_5 = self._calc_rsi(self.tdiff, 5)
        trsi_10 = self._calc_rsi(self.tdiff, 10)
        trsi_30 = self._calc_rsi(self.tdiff, 30)
        trsi_60 = self._calc_rsi(self.tdiff, 60)
        trsi_120 = self._calc_rsi(self.tdiff, 120)
        trsi_360 = self._calc_rsi(self.tdiff, 360)
        trsi_720 = self._calc_rsi(self.tdiff, 720)
        trsi_1440 = self._calc_rsi(self.tdiff, 1440)

        self.trsi = pd.concat((trsi_5, trsi_10, trsi_30,
                               trsi_60, trsi_120, trsi_360,
                               trsi_720, trsi_1440), axis=1)

        # Price Momentum 계산
        pm_5 = self.returns.rolling(5).sum()
        pm_10 = self.returns.rolling(10).sum()
        pm_30 = self.returns.rolling(30).sum()
        pm_60 = self.returns.rolling(60).sum()
        pm_120 = self.returns.rolling(120).sum()
        pm_360 = self.returns.rolling(360).sum()
        pm_720 = self.returns.rolling(720).sum()
        pm_1440 = self.returns.rolling(1440).sum()

        self.pm = pd.concat((pm_5, pm_10, pm_30,
                             pm_60, pm_120, pm_360,
                             pm_720, pm_1440), axis=1)

        # Return Volatility
        vol_5 = self.returns.rolling(5).std()
        vol_10 = self.returns.rolling(10).std()
        vol_30 = self.returns.rolling(30).std()
        vol_60 = self.returns.rolling(60).std()
        vol_120 = self.returns.rolling(120).std()
        vol_360 = self.returns.rolling(360).std()
        vol_720 = self.returns.rolling(720).std()
        vol_1440 = self.returns.rolling(1440).std()

        self.vol = pd.concat((vol_5, vol_10, vol_30,
                              vol_60, vol_120, vol_360,
                              vol_720, vol_1440), axis=1)

        # Turnover Volatility
        tvol_5 = self.treturns.rolling(5).std()
        tvol_10 = self.treturns.rolling(10).std()
        tvol_30 = self.treturns.rolling(30).std()
        tvol_60 = self.treturns.rolling(60).std()
        tvol_120 = self.treturns.rolling(120).std()
        tvol_360 = self.treturns.rolling(360).std()
        tvol_720 = self.treturns.rolling(720).std()
        tvol_1440 = self.treturns.rolling(1440).std()

        self.tvol = pd.concat((tvol_5, tvol_10, tvol_30,
                               tvol_60, tvol_120, tvol_360,
                               tvol_720, tvol_1440), axis=1)

        self.min_beg = 2 * 24 * 60 - 1
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
            reward = 1.
        elif ret > 0 and weights[0] < weights[1]:
            reward = -1.
        elif ret <= 0 and weights[0] >= weights[1]:
            reward = -1.
        elif ret <= 0 and weights[0] < weights[1]:
            reward = 1.

        next_obs = self._calc_obs()

        if self.cur_pos == self.end_pos:
            done = True
        else:
            done = False

        return next_obs, reward, ret, latest_w_prev, done

    def _calc_obs(self, eps=1e-6):
        " calculate observation "

        prsi = self.prsi.iloc[self.cur_pos]
        prsi = torch.FloatTensor(prsi.values.astype(float))
        prsi = prsi.unsqueeze(0)

        trsi = self.trsi.iloc[self.cur_pos]
        trsi = torch.FloatTensor(trsi.values.astype(float))
        trsi = trsi.unsqueeze(0)

        pm = self.pm.iloc[self.cur_pos]
        pm = torch.FloatTensor(pm.values.astype(float))
        pm = pm.unsqueeze(0)

        vol = self.vol.iloc[self.cur_pos]
        vol = torch.FloatTensor(vol.values.astype(float))
        vol = vol.unsqueeze(0)

        tvol = self.tvol.iloc[self.cur_pos]
        tvol = torch.FloatTensor(tvol.values.astype(float))
        tvol = tvol.unsqueeze(0)

        return [prsi, trsi, pm, vol, tvol]

    def _calc_rsi(self, pdiff, period=120):
        """
            RSI 계산
        """
        ppos = pdiff > 0.
        pneg = pdiff <= 0.

        dpos = (ppos * pdiff).rolling(period).sum()
        dneg = abs(pneg * pdiff).rolling(period).sum()

        rsi = dpos / (dpos + dneg)
        rsi = rsi.applymap(lambda x: 0. if pd.isnull(x) else x)

        return rsi
