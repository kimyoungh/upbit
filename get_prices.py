"""
    과거 시세 가져오는 스크립트
"""
import datetime
import numpy as np
import pandas as pd

from iqa_common.db_execute_manager import DBExecuteManager
from quotation import Quotation

time_format = '%Y-%m-%d %H:%M:%S'
outtime_format = '%Y-%m-%dT%H:%M:%S'

dbm = DBExecuteManager(host='localhost', port=3306)
col_query = "show columns from quantDB.bitcoin_price"
cols = dbm.get_fetchall(col_query)
cols = np.array(cols)[:, 0].ravel()
post_query = dbm.create_insert_query("quantDB.bitcoin_price", cols)

quo = Quotation()

limit = datetime.datetime(2019, 3, 6, 21, 0)

cnt = 0

while True:
    now = datetime.datetime.now()
    now_utc = quo._convert_kst_to_utc(now)

    limit_str = limit.strftime(time_format)

    returns = quo.get_market_price(period='minutes', to=limit_str,
                                   count=200)
    returns = returns.applymap(lambda x: 0 if pd.isnull(x) else x)

    dbm.set_commit(post_query, returns.values.tolist(), is_many=True)

    end_time = returns['candle_date_time_utc'].iloc[-1]
    end_time = datetime.datetime.strptime(end_time, outtime_format)

    end_time = datetime.datetime(end_time.year, end_time.month, end_time.day,
                                 end_time.hour, end_time.minute)
    now_utc = datetime.datetime(now_utc.year, now_utc.month, now_utc.day,
                                now_utc.hour, now_utc.minute)

    if end_time >= now_utc:
        break
    else:
        limit = limit + datetime.timedelta(minutes=200)
    print(limit)
