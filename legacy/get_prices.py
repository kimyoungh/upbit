"""
    과거 시세 가져오는 스크립트
"""
import pickle
import datetime
import numpy as np
import pandas as pd

from quotation import Quotation

time_format = '%Y-%m-%d %H:%M:%S'
outtime_format = '%Y-%m-%dT%H:%M:%S'

quo = Quotation()

pdata = pd.read_csv('bitcoin_price.csv', header=0, index_col=0)
pdata = pdata.set_index('candle_date_time_utc')

limit = datetime.datetime.strptime(pdata.index[-1], outtime_format)
limit = limit + datetime.timedelta(minutes=200)

cnt = 0

while True:
    now = datetime.datetime.now()
    now_utc = quo._convert_kst_to_utc(now)

    limit_str = limit.strftime(time_format)

    returns = quo.get_market_price(period='minutes', to=limit_str,
                                   count=200)
    returns = returns.applymap(lambda x: 0 if pd.isnull(x) else x)
    returns = returns.set_index('candle_date_time_utc')

    pdata = pdata.append(returns)
    pdata = pdata[~pdata.index.duplicated(keep='first')]

    with open('bitcoin_price_recent.pkl', 'wb') as f:
        pickle.dump(pdata, f)

    end_time = pdata.index[-1]
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
