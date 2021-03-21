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
             }


class Environ(gym.Env):
    " Environ for bitcoin trading "
    def __init__(self, config):
        self.config = config
        self.orderbooks = env_config['orderbooks']
        self.recent = env_config['recent']

    def reset(self):
        pass

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
        self.rec_cols = None
        self.index = None

        self.dbm = DBExecuteManager(host=host,
                                    port=port)

        self._get_order_cols()
        self._get_rec_cols()
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
                        {cols}
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

    def _get_rec_cols(self):
        query = "show columns from quantdb.crypto_recent_price"
        rec_cols = self.dbm.get_fetchall(query)
        rec_cols = np.array(rec_cols)[:, 0].ravel()

        self.rec_cols = rec_cols

    def _get_timestamp(self):
        " Get timestamp "
        order_query = """
                         select
                            distinct timestamp
                         from
                            quantdb.crypto_orderbooks
                         where
                            timestamp > 0
                      """
        rec_query = """
                        select
                            distinct timestamp
                        from
                            quantdb.crypto_recent_price
                        where
                            timestamp > 0
                    """

        order_t = self.dbm.get_fetchall(order_query)
        rec_t = self.dbm.get_fetchall(rec_query)

        order_t = pd.Index(pd.DataFrame(order_t).values.ravel())
        rec_t = pd.Index(pd.DataFrame(rec_t).values.ravel())

        index = order_t.append(rec_t).unique().sort_values()

        self.index = index
