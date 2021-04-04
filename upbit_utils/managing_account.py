"""
    업비트 계좌 관리 관련 모듈

    @author: Younghyun Kim
    Created on 2020.10.25
"""
import os
import uuid
import hashlib
import jwt
from urllib.parse import urlencode

import requests


class AccountManager:
    """ Managing Account """
    def __init__(self):
        self.access_key = os.environ['UPBIT_OPEN_API_ACCESS_KEY']
        self.secret_key = os.environ['UPBIT_OPEN_API_SECRET_KEY']
        self.server_url = os.environ['UPBIT_OPEN_API_SERVER_URL']

    def checking_account(self):
        " 전체 계좌 조회 "
        payload = {
            'access_key': self.access_key,
            'nonce': str(uuid.uuid4()),
            }

        jwt_token = jwt.encode(payload, self.secret_key)
        authorize_token = 'Bearer {}'.format(jwt_token)
        headers = {'Authorization': authorize_token}

        res = requests.get(self.server_url + "/v1/accounts", headers=headers)

        return res.json()

    def get_order_available_info(self, market='KRW-BTC'):
        """
            주문 가능 정보 조회

            Args:
                market: 조회 시장(default: KRW-BTC)

            Return:
                res: json
        """
        query = {
            'market': market
        }
        query_string = urlencode(query).encode()

        m = hashlib.sha512()
        m.update(query_string)
        query_hash = m.hexdigest()

        payload = self._get_payload(query_hash, 'SHA512')

        jwt_token = jwt.encode(payload, self.secret_key)
        authorize_token = 'Bearer {}'.format(jwt_token)
        headers = {"Authorization": authorize_token}

        res = requests.get(self.server_url + '/v1/orders/chance',
                           params=query, headers=headers)

        return res.json()

    def get_each_order_info(self, order_uuid):
        """
            개별 주문 조회: order_uuid를 통해 개별 주문건을 조회한다.

            Args:
                order_uuid: 주문 uuid
            Return:
                res: json
        """
        query = {
            'uuid': order_uuid,
        }
        query_string = urlencode(query).encode()

        m = hashlib.sha512()
        m.update(query_string)
        query_hash = m.hexdigest()

        payload = self._get_payload(query_hash, 'SHA512')

        jwt_token = jwt.encode(payload, self.secret_key)
        authorize_token = 'Bearer {}'.format(jwt_token)
        headers = {"Authorization": authorize_token}

        res = requests.get(self.server_url + "/v1/order", params=query,
                           headers=headers)

        return res.json()

    def get_order_list(self, uuids, state,
                       market='KRW-BTC', kind='normal',
                       page=1, limit=100, order_by='desc'):
        """
            주문 리스트 조회
            Args:
                uuids: list. 주문 고유 아이디 목록
                state: 주문 상태
                        - wait: 체결 대기(default)
                        - done: 전체 체결 완료
                        - cancel: 주문 취소
                market: market(default: KRW-BTC)
                kind: 주문 유형
                        - normal: 일반 주문
                        - watch: 예약 주문
                page: 페이지 수
                limit: 요청 개수
                order_by: 정렬 방식
                            - asc: 오름차순
                            - desc: 내림차순
            Return:
                res: json
        """
        query = {
            'state': state,
        }
        query_string = urlencode(query)

        uuids_query_string = '&'.join(["uuids[]={}".format(uid) for uid in uuids])

        query['uuids[]'] = uuids
        query_string = "{0}&{1}".format(query_string, uuids_query_string).encode()

        m = hashlib.sha512()
        m.update(query_string)
        query_hash = m.hexdigest()

        payload = self._get_payload(query_hash, 'SHA512')

        jwt_token = jwt.encode(payload, self.secret_key)
        authorize_token = 'Bearer {}'.format(jwt_token)
        headers = {'Authorization': authorize_token}

        res = requests.get(self.server_url + "/v1/orders", params=query,
                           headers=headers)

        return res.json()

    def cancel_order(self, order_uuid):
        """
            주문 uuid를 통해 해당 주문에 대한 취소 접수를 한다.

            Args:
                order_uuid: urder uuid
            Return:
                res: json
        """
        query = {
            'uuid': order_uuid,
        }
        query_string = urlencode(query).encode()

        m = hashlib.sha512()
        m.update(query_string)
        query_hash = m.hexdigest()

        payload = self._get_payload(query_hash, 'SHA512')

        jwt_token = jwt.encode(payload, self.secret_key)
        authorize_token = 'Bearer {}'.format(jwt_token)
        headers = {'Authorization': authorize_token}

        res = requests.delete(self.server_url + "/v1/order", params=query,
                              headers=headers)

        return res.json()

    def order(self, side, volume, price, ord_type, market='KRW-BTC'):
        """ 주문 요청을 한다.

            Args:
                side: str. 주문 종류(필수)
                            - bid: 매수
                            - ask: 매도
                volume: float. 주문량(지정가, 시장가 매도 시 필수)
                price: float. 주문 가격(지정가, 시장가 매수 시 필수)
                ord_type: str. 주문 타입(필수)
                                - limit: 지정가 주문
                                - price: 시장가 주문(매수)
                                - market: 시장가 주문(매도)
                market: market(default: KRW-BTC)
        """
        query = {
            'market': market,
            'side': side,
            'volume': str(volume) if volume is not None else '',
            'price': str(price) if price is not None else '',
            'ord_type': ord_type,
        }
        query_string = urlencode(query).encode()

        m = hashlib.sha512()
        m.update(query_string)
        query_hash = m.hexdigest()

        payload = self._get_payload(query_hash, 'SHA512')

        jwt_token = jwt.encode(payload, self.secret_key)
        authorize_token = 'Bearer {}'.format(jwt_token)
        headers = {'Authorization': authorize_token}

        res = requests.post(self.server_url + "/v1/orders", params=query,
                            headers=headers)

        return res.json()

    def _get_payload(self, query_hash, query_hash_alg='SHA512'):
        payload = {
            'access_key': self.access_key,
            'nonce': str(uuid.uuid4()),
            'query_hash': query_hash,
            'query_hash_alg': query_hash_alg,
        }

        return payload
