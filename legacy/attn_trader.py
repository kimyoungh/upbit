"""
    업비트 계좌와 light allocator를 이용한 트레이딩 모듈 + Transformer

    @author: Younghyun Kim
    Created on 2021.01.01
"""
import os
import pickle
import time
import datetime
import numpy as np
import pandas as pd

import torch

from quotation import Quotation
from attn_allocator import BitInvestor
from managing_account import AccountManager

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

MODEL = "./models/attn.pt"

MODEL_CONFIG = {
    'prsi_dim': 32,
    'trsi_dim': 8,
    'pm_dim': 8,
    'vol_dim': 8,
    'tvol_dim': 8,
    'hidden_dim': 4,
    'nheads': 2,
    'dropout': 0.1
}


class BitTrader:
    """ Bitcoin Trader """
    def __init__(self, model_config=MODEL_CONFIG,
                 model_path=MODEL, device='cuda:0',
                 price_path='./bitcoin_price_recent.pkl',
                 trading_log_path='./trading_log.txt'):
        " Initialization "
        self.model_config = model_config
        self.model_path = model_path

        if torch.cuda.is_available() and device != 'cpu':
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        self.price_path = price_path

        self.trader = BitInvestor(model_config['prsi_dim'],
                                  model_config['trsi_dim'],
                                  model_config['pm_dim'],
                                  model_config['vol_dim'],
                                  model_config['tvol_dim'],
                                  model_config['hidden_dim'],
                                  model_config['nheads'],
                                  model_config['dropout'])
        self.trader = self.trader.to(self.device)
        self.trader.load_state_dict(torch.load(model_path,
                                               map_location=self.device))
        self.trader.eval()

        self.quo = Quotation()
        self.account = AccountManager()

        self.price = None
        self.pdata = None
        self.turnover = None
        self.returns = None
        self.treturns = None
        self.pdiff = None
        self.tdiff = None

        self.prsi = None
        self.trsi = None
        self.pm = None
        self.vol = None
        self.tvol = None
        self.trading_log = open(trading_log_path, 'a')

        with open(price_path, 'rb') as f:
            self.price = pickle.load(f)

        self.now = datetime.datetime.now()

        self.time_format = "%Y-%m-%d %H:%M:%S"
        self.outtime_format = "%Y-%m-%dT%H:%M:%S"

        self.uuids = []

    def get_now(self):
        self.now = datetime.datetime.now()

    def get_price(self, m=200):
        limit = datetime.datetime.strptime(self.price.index[-1],
                                           self.outtime_format)
        limit = limit + datetime.timedelta(minutes=m)

        while True:
            self.get_now()
            now_utc = self.quo._convert_kst_to_utc(self.now)

            limit_str = limit.strftime(self.time_format)

            price = self.quo.get_market_price(period='minutes', to=limit_str,
                                              count=m)
            price = price.applymap(lambda x: 0 if pd.isnull(x) else x)
            price = price.set_index('candle_date_time_utc')

            self.price = self.price.append(price)
            self.price = self.price[~self.price.index.duplicated(keep='first')]

            with open(self.price_path, 'wb') as f:
                pickle.dump(self.price, f)

            end_time = self.price.index[-1]
            end_time = datetime.datetime.strptime(end_time, self.outtime_format)

            end_time = datetime.datetime(end_time.year, end_time.month,
                                         end_time.day, end_time.hour,
                                         end_time.minute)
            now_utc = datetime.datetime(now_utc.year, now_utc.month, now_utc.day,
                                        now_utc.hour, now_utc.minute)

            if end_time >= now_utc:
                break
            limit = limit + datetime.timedelta(minutes=m)

    def check_and_cancel_order_uuid(self):
        " check order uuid "
        if len(self.uuids) > 0:
            for uuid in self.uuids:
                order = self.account.get_each_order_info(uuid)

                if order['state'] == 'wait':
                    self.account.cancel_order(uuid)

            self.uuids.clear()

    def get_obs(self):
        " calculate observation "
        self._calc_features()
        prsi = self.prsi.iloc[-1]
        prsi = torch.FloatTensor(prsi.values.astype(float))
        prsi = prsi.unsqueeze(0).to(self.device)

        trsi = self.trsi.iloc[-1]
        trsi = torch.FloatTensor(trsi.values.astype(float))
        trsi = trsi.unsqueeze(0).to(self.device)

        pm = self.pm.iloc[-1]
        pm = torch.FloatTensor(pm.values.astype(float))
        pm = pm.unsqueeze(0).to(self.device)

        vol = self.vol.iloc[-1]
        vol = torch.FloatTensor(vol.values.astype(float))
        vol = vol.unsqueeze(0).to(self.device)

        tvol = self.tvol.iloc[-1]
        tvol = torch.FloatTensor(tvol.values.astype(float))
        tvol = tvol.unsqueeze(0).to(self.device)

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

    def _calc_features(self, eps=1e-6):
        " calculate features "
        self.pdata = self.price[['opening_price', 'high_price',
                                 'low_price', 'trade_price']]
        self.turnover = self.price[['candle_acc_trade_price']]
        self.pdiff = self.pdata.iloc[-1441:].diff(1)
        self.tdiff = self.turnover.iloc[-1441:].diff(1)

        self.returns = np.log(self.price['trade_price']).diff(1)
        self.treturns = np.log(self.price['candle_acc_trade_price'] +
                               eps).diff(1)

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

    def trade(self, market='KRW-BTC', down_limit=-0.1, eps=10000.0,
              min_v=1000.0):
        "Trading Method"
        cnt = 0
        data_length = self.price.shape[0]
        uuid = None
        init_nav = None
        nav = None

        info = self.account.get_order_available_info()
        bid_fee = float(info['bid_fee'])
        ask_fee = float(info['ask_fee'])

        while True:
            self.get_now()
            now = self.quo._convert_kst_to_utc(self.now)
            kor_now = self.now.strftime(self.time_format)

            if uuid is not None:
                res = self.account.get_each_order_info(uuid)
                state = res['state']
                price = res['price']
                volume = res['volume']

                if state == 'done':
                    print(res)
                    print(kor_now, state, price, volume)
                uuid = None

            # 최근 데이터가 없는 경우 가져오기
            if self.price.index[-1] < now.strftime(self.outtime_format):
                self.get_price()

            self.trader.load_state_dict(torch.load(self.model_path,
                                                   map_location=self.device))
            self.trader.eval()
            # 직전 체결가 가져와서 현재 순자산 계산
            res = self.quo.get_recent_traded_data(market=market, count=1,
                                                  daysAgo=0)
            rec_price = res[0]['trade_price']

            balance = self.account.checking_account()

            bal_btc = float(balance[0]['balance'])
            btc_value = rec_price * bal_btc
            krw_value = float(balance[1]['balance'])

            nav = btc_value + krw_value
            self.trading_log.write(kor_now + "\t" + "NAV: " +
                                   str(nav) + "\n")

            # 트레이딩 초기 순자산 저장
            if cnt == 0:
                init_nav = nav
                print(btc_value)
                print(krw_value)
                print(nav)

            cumreturn = (nav / init_nav) - 1.
            if cumreturn <= down_limit:
                self.trading_log.write("Downside_limit_over " +
                                       kor_now + "\t" +
                                       str(cumreturn) + "\n")
                break

            # 데이터 업데이트가 된 경우에만 트레이딩
            if data_length < self.price.shape[0]:
                data_length = self.price.shape[0]

                # 최근 주문 내역 취소
                self.check_and_cancel_order_uuid()

                btc_prob = btc_value / nav
                krw_prob = krw_value / nav

                # 최신 투자비중
                weights_rec = torch.FloatTensor([[btc_prob,
                                                  krw_prob]]).to(self.device)
                obs = self.get_obs()

                # 리밸런싱 의사결정
                print(obs[0])
                rweights, _, _, rebal = self.trader(*obs, weights_rec,
                                                    train=False, init=False)

                rebal = rebal.item()
                print(rebal)
                if rebal == 1.:
                    print("weights_rec: ", weights_rec)
                    print("rweights: ", rweights)
                    btc_w = rweights[0, -2].item()
                    btc_diff = btc_w - btc_prob

                    if btc_diff == 0.:
                        continue

                    if btc_diff > 0:
                        tnav = nav * (1. - bid_fee) - eps
                    else:
                        tnav = nav * (1. - ask_fee) - eps
                    btc_trading_bal = tnav * btc_diff / rec_price

                    if btc_trading_bal > 0:
                        side = 'bid'
                        ord_type = 'price'
                        b_price = rec_price * btc_trading_bal
                        btc_bal = None
                    else:
                        side = 'ask'
                        ord_type = 'market'
                        b_price = None
                        btc_bal = abs(btc_trading_bal)

                    btc_trading_bal = abs(btc_trading_bal)

                    if btc_trading_bal * rec_price >= min_v:
                        # 주문하기
                        res = self.account.order(side, btc_bal,
                                                 b_price, ord_type,
                                                 market=market)
                        print(side)
                        print(res)
                        uuid = res['uuid']

            cnt += 1

if __name__ == "__main__":
    trader = BitTrader(device='cuda:0')

    trader.trade()
