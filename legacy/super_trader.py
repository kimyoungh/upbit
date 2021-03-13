"""
    업비트 계좌와 super allocator를 이용한 트레이딩 모듈

    @author: Younghyun Kim
    Created on 2021.01.11
"""
import os
import pickle
import time
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from quotation import Quotation
from super_allocator import SuperInvestor
from managing_account import AccountManager

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

COLUMNS = ['opening_price', 'high_price', 'low_price', 'trade_price',
           'candle_acc_trade_price', 'candle_acc_trade_volume']
MODEL = "./super_trader.pt"

MODEL_CONFIG = {
    'input_dim': 6,
    'seq_length': 1440,
    'emb_dim': 4,
    'hidden_dim': 16,
    'nheads': 2,
    'dropout': 0.1
}


class SuperTrader:
    """ Bitcoin Super Trader """
    def __init__(self, model_config=MODEL_CONFIG, cols=COLUMNS,
                 model_path=MODEL, device='cuda:0',eps=1e-6,
                 price_path='./bitcoin_price_recent.pkl',
                 trading_log_path='./super_trading_log.txt'):
        " Initialization "
        self.model_config = model_config
        self.model_path = model_path

        if torch.cuda.is_available() and device != 'cpu':
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        self.price_path = price_path
        self.softmax = nn.Softmax(dim=1)

        self.trader = SuperInvestor(model_config['input_dim'],
                                    model_config['seq_length'],
                                    model_config['emb_dim'],
                                    model_config['hidden_dim'],
                                    model_config['nheads'],
                                    model_config['dropout'])
        self.trader = self.trader.to(device)
        self.trader.load_state_dict(torch.load(model_path,
                                               map_location=device))
        self.trader.eval()

        self.seq_length = model_config['seq_length']
        self.cols = cols
        self.eps = eps

        self.quo = Quotation()
        self.account = AccountManager()

        self.price = None
        self.pdata = None

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
        pdata = self.price.iloc[-self.seq_length:].copy()[self.cols].values
        pdata = pdata / (pdata[0] + self.eps) - 1.
        pdata = torch.FloatTensor(pdata).unsqueeze(0).to(self.device)

        return pdata

    def trade(self, market='KRW-BTC', down_limit=-0.1, eps=10000.0,
              min_v=10000.0):
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

            # 직전 체결가 가져와서 현재 순자산 계산
            res = self.quo.get_recent_traded_data(market=market, count=1,
                                                  daysAgo=0)
            rec_price = res[0]['trade_price']

            balance = self.account.checking_account()
            print(balance)

            if len(balance) > 1 and balance[1]['currency'] == 'KRW':
                bal_btc = float(balance[0]['balance'])
                btc_value = rec_price * bal_btc
                krw_value = float(balance[1]['balance'])
            else:
                if balance[0]['currency'] == 'KRW':
                    btc_value = 0.
                    krw_value = float(balance[0]['balance'])
                else:
                    bal_btc = float(balance[0]['balance'])
                    btc_value = rec_price * bal_btc
                    krw_value = 0.

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
                with torch.no_grad():
                    pred = self.trader(obs)

                pred = pred.detach()
                print(pred)

                """
                pred += abs(pred.min())
                pred /= pred.sum()
                """
                pred = self.softmax(pred)
                pred = pred.numpy()

                # 매매 판단
                print("weights_rec: ", weights_rec)
                print("pred: ", pred)
                btc_w = pred[0, 0]
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
            self.trader.load_state_dict(torch.load(self.model_path,
                                                   map_location=self.device))
            self.trader.eval()

if __name__ == "__main__":
    trader = SuperTrader(device='cpu')

    trader.trade()
