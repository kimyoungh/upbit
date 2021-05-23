"""
    Environment module for bitcoin trading with orderbooks

    Created on 2021.03.21
    @author: Younghyun Kim
"""
import numpy as np
import pandas as pd
import gym

from iqa_common.db_execute_manager import DBExecuteManager
from iqa_common.constants import QUANT_DB


env_config = {
                'orderbooks': None,
                'recent': None,
                'fee': 0.005,
                'min_episode_length': 20000,
                'window': 100,
                'ask_price_cols': np.array(
                        ['ask_price_'+str(i)
                            for i in np.arange(1, 16)]),
                'ask_size_cols': np.array(
                        ['ask_size_'+str(i)
                            for i in np.arange(1, 16)]),
                'bid_price_cols': np.array(
                        ['bid_price_'+str(i)
                            for i in np.arange(1, 16)]),
                'bid_size_cols': np.array(
                        ['bid_size_'+str(i)
                            for i in np.arange(1, 16)]),
                'downside_limit': -0.05,
             }


class Environ(gym.Env):
    " Environ for bitcoin trading "
    def __init__(self, config=env_config):
        self.config = config

        observ = pd.concat((config['orderbooks'],
                            config['recent']), axis=1)
        observ['trade_price'] = observ['trade_price'].bfill()
        observ = observ.dropna()
        self.observ = observ

        self.ask_price_cols = config['ask_price_cols']
        self.ask_size_cols = config['ask_size_cols']
        self.bid_price_cols = config['bid_price_cols']
        self.bid_size_cols = config['bid_size_cols']

        self.index = self.observ.index

        self.fee = config['fee']

        self.min_episode_length =\
            config['min_episode_length']
        self.window = config['window']

        self.downside_limit = config['downside_limit']  # 손실 허용 범위

        self.epi_index = None
        self.cur_pos = None
        self.end_pos = None

        self.orderbook = None
        self.total_bid_ask = None
        self.trade_price = None
        self.asset = 1.
        self.possession = False
        self.possession_series = np.zeros(self.window)

    def reset(self, min_episode_length=None):
        """
            Reset episodes
            Args:
                min_episode_length(default: self.min_episode_length)

            Return:
                obs: orderbook, total_bid_ask,
                     trade_price, possession_series
                     * trade_price: 한 시점 씩 앞당겨짐
        """
        if min_episode_length is None:
            min_epi = self.min_episode_length
        else:
            min_epi = min_episode_length

        start_pos = np.random.choice(np.arange(self.window - 1,
                                               self.index.shape[0] -
                                               min_epi + 1),
                                     1, replace=False).item()

        end_t = self.index[start_pos + min_epi]

        end_pos = np.argwhere(self.index == end_t).item()

        self.epi_index = self.index[start_pos:end_pos+1]
        self.cur_pos = self.window - 1
        self.end_pos = self.epi_index.shape[0] - 1

        self.possession = False

        orderbook, total_bid_ask, trade_price, possession_series =\
            self.calc_obs()

        # Normalize
        orderbook[0][:, 0] =\
            self.normalize_series(orderbook[0][:, 0])
        orderbook[0][:, 1] =\
            self.normalize_series(orderbook[0][:, 1])
        total_bid_ask = self.normalize_series(total_bid_ask)
        trade_price = self.normalize_series(trade_price)

        self.asset = 1.

        return [orderbook, total_bid_ask,
                trade_price, possession_series]

    def step(self, action, mul=100., neg=-1.,
             rmin=-1., rmax=+1.):
        """
            action: [0: sell, 1: hold, 2: buy]
        """
        self.cur_pos += 1

        orderbook, total_bid_ask, trade_price, possession_series =\
            self.calc_obs()

        ret = trade_price[-1] / trade_price[-2] - 1.

        if self.possession:
            self.asset *= (1. + ret)

        if self.cur_pos == self.end_pos or\
           self.asset <= (1. - self.downside_limit):
            done = True
        else:
            done = False

        if action == 0:
            if self.possession:
                cum_ret = -ret
                cum_ret -= self.fee
                reward = cum_ret * mul

                self.asset -= self.fee

                self.possession = False
            else:
                reward = neg
        elif action == 1:
            if self.possession:
                reward = ret * mul
            else:
                reward = -ret * mul
        elif action == 2:
            if not self.possession:
                cum_ret = ret - self.fee
                reward = cum_ret * mul

                self.possession = True
                self.asset *= (1. + cum_ret)
            else:
                reward = neg

        reward = np.clip(reward, rmin, rmax)

        # Normalize
        orderbook[0][:, 0] =\
            self.normalize_series(orderbook[0][:, 0])
        orderbook[0][:, 1] =\
            self.normalize_series(orderbook[0][:, 1])
        total_bid_ask = self.normalize_series(total_bid_ask)
        trade_price = self.normalize_series(trade_price)

        return [orderbook, total_bid_ask,
                trade_price, possession_series],\
            reward, done

    def calc_obs(self):
        " calc observation "
        observ = self.observ.loc[
                self.epi_index[
                    self.cur_pos-(self.window-1):self.cur_pos+1]]
        ask_price = observ[self.ask_price_cols].values[:, ::-1]
        ask_size = observ[self.ask_size_cols].values[:, ::-1]
        bid_price = observ[self.bid_price_cols].values
        bid_size = observ[self.bid_size_cols].values
        total_ask_size = observ['total_ask_size'].values
        total_bid_size = observ['total_bid_size'].values
        trade_price = observ['trade_price'].values

        price = np.concatenate((ask_price, bid_price), axis=1)
        size = np.concatenate((ask_size, bid_size), axis=1)

        orderbook =\
            np.hstack((price, size)).reshape(self.window, 2, -1)

        total_bid_ask =\
            np.vstack((total_ask_size, total_bid_size)).transpose()

        self.possession_series[:-1] = self.possession_series[1:]
        self.possession_series[-1] = float(self.possession)

        return orderbook, total_bid_ask,\
            trade_price, self.possession_series

    @staticmethod
    def normalize_series(series):
        """ normalize series data """
        smax = series.max()
        smin = series.min()

        if smax == smin:
            normalized = np.zeros_like(series)
        else:
            normalized = (series - smin) / (smax - smin)

        return normalized


class DataLoader:
    " Data Loader "
    def __init__(self, host=QUANT_DB['host'],
                 port=QUANT_DB['port'],
                 start_t=None, end_t=None):
        """
            Args:
                start_t: timestamp start time
                end_t: timestamp end time
        """
        self.start_t = start_t
        self.end_t = end_t
        self.host = host
        self.port = port

        self.order_cols = None
        self.rec_cols = np.array(['timestamp', 'trade_price'])
        self.index = None

        self.dbm = DBExecuteManager(host=host,
                                    port=port)

        self._get_order_cols()
        self._get_timestamp()

    def get_orderbooks(self):
        " get orderbooks "
        query = """
                    select
                        {cols}
                    from
                        quantdb.crypto_orderbooks
                    where
                        timestamp between
                        {start} and {end}
                """
        if self.start_t is None:
            start_t = self.index[0]
        else:
            start_t = self.start_t

        if self.end_t is None:
            end_t = self.index[-1]
        else:
            end_t = self.end_t

        query = query.format(cols=",".join(self.order_cols),
                             start=start_t, end=end_t)
        orderbooks = self.dbm.get_fetchall(query)
        orderbooks = pd.DataFrame(orderbooks,
                                  columns=self.order_cols)
        orderbooks =\
            orderbooks.drop_duplicates('timestamp', keep='last')
        orderbooks = orderbooks.set_index('timestamp')

        orderbooks = orderbooks.reindex(index=self.index)

        return orderbooks

    def get_recent_price(self):
        " get recent price "
        query = """
                    select
                        timestamp, trade_price
                    from
                        quantdb.crypto_recent_price
                    where
                        timestamp between
                        {start} and {end}
                """
        if self.start_t is None:
            start_t = self.index[0]
        else:
            start_t = self.start_t

        if self.end_t is None:
            end_t = self.index[-1]
        else:
            end_t = self.end_t

        query = query.format(cols=",".join(self.rec_cols),
                             start=start_t, end=end_t)
        recent = self.dbm.get_fetchall(query)
        recent = pd.DataFrame(recent, columns=self.rec_cols)

        recent = recent.drop_duplicates('timestamp', keep='last')
        recent = recent.set_index('timestamp')

        recent = recent.reindex(index=self.index)

        return recent

    def _get_order_cols(self):
        query = "show columns from quantdb.crypto_orderbooks"
        order_cols = self.dbm.get_fetchall(query)
        order_cols = np.array(order_cols)[:, 0].ravel()

        self.order_cols = order_cols

    def _get_timestamp(self):
        " Get timestamp "
        order_query = """
                         select
                            distinct timestamp
                         from
                            quantdb.crypto_orderbooks
                         where
                            timestamp between
                            {start_t} and {end_t}
                      """
        rec_query = """
                        select
                            distinct timestamp
                        from
                            quantdb.crypto_recent_price
                        where
                            timestamp between
                            {start_t} and {end_t}
                    """
        if self.start_t is None:
            start_t = self.index[0]
        else:
            start_t = self.start_t

        if self.end_t is None:
            end_t = self.index[-1]
        else:
            end_t = self.end_t

        order_query = order_query.format(start_t=start_t,
                                         end_t=end_t)
        rec_query = rec_query.format(start_t=start_t,
                                     end_t=end_t)

        order_t = self.dbm.get_fetchall(order_query)
        rec_t = self.dbm.get_fetchall(rec_query)

        order_t = pd.Index(pd.DataFrame(order_t).values.ravel())
        rec_t = pd.Index(pd.DataFrame(rec_t).values.ravel())

        index = order_t.append(rec_t).unique().sort_values()

        self.index = index
