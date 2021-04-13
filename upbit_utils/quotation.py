"""
    업비트 시세 조회 모듈

    @author: Younghyun Kim
    Created on: 2020.10.25
"""
import datetime
import requests
import pandas as pd


class Quotation:
    " Quotation Class "
    def __init__(self):
        self.url = "https://api.upbit.com"

    def get_market_code(self, details=False):
        """
            마켓 코드 조회

            Args:
                details: 유의종목 필드와 같은 상세 정보 노출(default: False)

            Return:
                res: json
                     - market: 업비트에서 제공중인 시장 정보
                     - korean_name: 거래 대상 암호화폐 한글명
                     - english_name: 거래 대상 암호화폐 영문명
                     - market_warning: 유의 종목 여부
                                        - NONE(해당 사항 없음),
                                        - CAUTION(투자유의)
        """
        url = self.url + "/v1/market/all"

        querystring = {"isDetails": str(details).lower()}

        res = requests.request("GET", url, params=querystring)

        return res.json()

    def get_market_price(self, period, to=None, unit=1,
                         market='KRW-BTC', count=1):
        """
            시세 조회 메소드

            Args:
                period: str. 시간단위(minutes, days, weeks, months)
                to: str. 마지막 캔들 UTC 시각 ('yyyy-MM-dd HH:mm:ss')
                unit: int. period가 minute일때, 시간 단위(1: 1분)
                market: market
                count: int. 최대 조회 가격 수(최대 200)
            Return:
                returns: pd.DataFrame
        """
        url = self.url + "/v1/candles/" + period

        if period == 'minutes':
            url += "/" + str(unit)

        querystring = {
                        'market': market,
                        'count': str(count),
        }

        if to is not None:
            querystring['to'] = to

        response = requests.request('GET', url, params=querystring)

        returns = pd.DataFrame(response.json()).iloc[::-1]
        times = returns['candle_date_time_kst']
        times = times.apply(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S'))
        dates = times.apply(lambda x: x.date())
        hour = times.apply(lambda x: x.hour)
        minute = times.apply(lambda x: x.minute)
        second = times.apply(lambda x: x.second)

        returns['trade_date'] = dates
        returns['trade_hour'] = hour
        returns['trade_minute'] = minute
        returns['trade_second'] = second

        return returns

    def get_recent_traded_data(self, to=None, market='KRW-BTC',
                               count=1, daysAgo=1):
        """
            최근 체결 내역

            Args:
                to: str. HHmmss or HH:mm:ss 마지막 체결 시각
        """
        url = self.url + "/v1/trades/ticks"

        querystring = {'count': str(count),
                       'market': market,
                       'daysAgo': str(daysAgo)}

        if to is not None:
            querystring['to'] = to

        res = requests.request("GET", url, params=querystring)

        return res.json()

    def get_recent_price(self, market='KRW-BTC'):
        """
            현재가 정보
        """
        url = self.url + "/v1/ticker"
        query = {'markets': market}

        res = requests.request('GET', url, params=query)

        return res.json()

    def get_orderbook(self, markets='KRW-BTC'):
        """
             호가 정보 조회
        """
        url = self.url + "/v1/orderbook"

        query = {'markets': markets}

        res = requests.request('GET', url, params=query)

        return res.json()

    def _convert_kst_to_utc(self, date):
        """
            kst 시간을 utc로 변환하기
        """
        if isinstance(date, str):
            date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

        date += datetime.timedelta(hours=-9)

        return date
