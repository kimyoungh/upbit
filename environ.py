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
                'min_episode_length': 200000,
                'ask_price_cols': np.array(
                        ['ask_price_'+str(i)
                            for i in np.arange(1, 15)]),
                'ask_size_cols': np.array(
                        ['ask_size_'+str(i)
                            for i in np.arange(1, 15)]),
                'bid_price_cols': np.array(
                        ['bid_price_'+str(i)
                            for i in np.arange(1, 15)]),
                'bid_size_cols': np.array(
                        ['bid_size_'+str(i)
                            for i in np.arange(1, 15)]),
             }


class Environ(gym.Env):
    " Environ for bitcoin trading "
    def __init__(self, config):
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
            env_config['min_episode_length']

        self.epi_index = None
        self.cur_pos = None

    def reset(self, min_episode_length=None):
        start_pos = np.random.choice(np.arange(self.index.shape[0]),
                                     1,
                                     replace=False).item()
        """
            Reset episodes
            Args:
                min_episode_length(default: self.min_episode_length)

            Return:
                obs: ask_price, ask_size, bid_price,
                     bid_size, total_ask_size, total_bid_size,
                     trade_price
        """
        if min_episode_length is None:
            min_epi = self.min_episode_length
            end_t = np.random.choice(
                    self.index[start_pos + min_epi:],
                    1, replace=False).item()
        else:
            end_t = np.random.choice(
                    self.index[start_pos + min_episode_length-1:],
                    1, replace=False).item()
        end_pos = np.argwhere(self.index == end_t).item()

        self.epi_index = self.index[start_pos:end_pos+1]
        self.cur_pos = 0

        observ = self.observ.loc[self.epi_index[self.cur_pos]]
        ask_price = observ[self.ask_price_cols].values
        ask_size = observ[self.ask_size_cols].values
        bid_price = observ[self.bid_price_cols].values
        bid_size = observ[self.bid_size_cols].values
        total_ask_size = observ['total_ask_size']
        total_bid_size = observ['total_bid_size']
        trade_price = observ['trade_price']

        return [ask_price, ask_size, bid_price,
                bid_size, total_ask_size, total_bid_size,
                trade_price]

    def step(self, action):
        pass


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
