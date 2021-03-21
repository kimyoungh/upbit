"""
    Processing module for realtime crypto data

    @author: Younghyun Kim
    Created on 2021.03.14
"""
import traceback
import datetime
import numpy as np
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

from iqa_common.constants import QUANT_DB as qdb
from iqa_common.db_execute_manager import DBExecuteManager

from upbit_utils.quotation import Quotation

CONFIG = {'host': qdb['host'],
          'port': qdb['port'],
          'user': qdb['user'],
          'passwd': qdb['passwd'],
          'db': qdb['db'],
          'market': 'KRW-BTC',
          'interval': 0.1,  # interval seconds
          }


class RealtimeProcessor:
    " Realtime Processor "
    def __init__(self, config=CONFIG):

        self.rec_price_cols = None
        self.orderbooks_cols = None

        self.config = config
        self.market = config['market']
        self.interval = config['interval']

        self.quo = Quotation()

        self.dbm_1 = DBExecuteManager(host=config['host'],
                                      port=config['port'],
                                      user=config['user'],
                                      passwd=config['passwd'],
                                      db=config['db'])

        self.dbm_2 = DBExecuteManager(host=config['host'],
                                      port=config['port'],
                                      user=config['user'],
                                      passwd=config['passwd'],
                                      db=config['db'])

        self._get_rec_price_columns()
        self._get_orderbooks_columns()
        self._create_rec_price_insert_query()
        self._create_orderbooks_insert_query()

        self.sched = BlockingScheduler()

    def start(self):
        self.sched.add_job(self.processing_recent_price,
                           'interval', seconds=self.interval,
                           id='recent_price')
        self.sched.add_job(self.processing_orderbooks,
                           'interval', seconds=self.interval,
                           id='orderbooks')

        print('process beginning...')
        self.sched.start()

    def processing_recent_price(self):
        """
            processing recent price
                1. get recent price from quo
                2. upload to quantdb
        """
        # Get data
        rec_price = self.quo.get_recent_price(market=self.market)
        rec_price = pd.DataFrame(rec_price)[self.rec_price_cols]
        rec_price[['trade_timestamp', 'timestamp']] =\
            rec_price[['trade_timestamp', 'timestamp']].\
            applymap(lambda x: x / 1000.)

        rec_list = rec_price.values.tolist()
        self.dbm_1.set_commit(self.price_query, rec_list,
                            is_many=True)

    def processing_orderbooks(self):
        """
            processing orderbooks
                1. get orderbooks
                2. upload to quantdb
        """
        # Get data
        orders = self.quo.get_orderbook(markets=self.market)
        o_len = len(orders)

        for i in range(o_len):
            order_list = []
            for key, value in orders[i].items():
                if key != 'orderbook_units':
                    if key == 'timestamp':
                        value /= 1000.
                    order_list.append(value)
                else:
                    for units in value:
                        for _, uvalue in units.items():
                            order_list.append(uvalue)
            self.dbm_2.set_commit(self.order_query, [order_list],
                                is_many=True)

    def _create_rec_price_insert_query(self):
        table_name = 'quantdb.crypto_recent_price'

        self.price_query =\
            self.dbm_1.create_insert_query(table_name,
                                         self.rec_price_cols)

    def _create_orderbooks_insert_query(self):
        table_name = 'quantdb.crypto_orderbooks'

        self.order_query =\
            self.dbm_2.create_insert_query(table_name,
                                         self.orderbooks_cols)

    def _get_rec_price_columns(self):
        " get crypto recent price columns "
        query = """
                    show
                        columns
                    from quantdb.crypto_recent_price
                """
        res = self.dbm_1.get_fetchall(query)
        res = pd.DataFrame(res).values[:, 0].ravel()

        self.rec_price_cols = res

    def _get_orderbooks_columns(self):
        " get crypto orderbooks columns "
        query = """
                    show
                        columns
                    from quantdb.crypto_orderbooks
                """
        res = self.dbm_2.get_fetchall(query)
        res = pd.DataFrame(res).values[:, 0].ravel()

        self.orderbooks_cols = res


if __name__ == '__main__':
    rp = RealtimeProcessor()

    rp.start()
