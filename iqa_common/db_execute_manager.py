"""
    investmentQuantAI Database 관리 모듈

    Created on 2020.06.20.
    @Author: Younghyun Kim
"""

import pymysql

from iqa_common.constants import QUANT_DB as qdb

class DBExecuteManager:
    """
        investmentQuantAI DB Read/Write 관리 클래스
    """
    def __init__(self, host=qdb['host'], port=qdb['port'],
                 user=qdb['user'], passwd=qdb['passwd'], db=qdb['db']):
        " Initialization "
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.db = db

        self.quant_db = None

    def _set_db(self):
        """
            DB를 쓰거나 읽을 때마다 DB를 불러오는 함수
        """
        quant_db = pymysql.connect(host=self.host, port=self.port,
                                   user=self.user, passwd=self.passwd,
                                   db=self.db, charset='utf8')

        self.quant_db = quant_db

    def _db_close(self):
        " Closing DB function "
        self.quant_db.close()
        self.quant_db = None

    def create_table(self, query):
        """
            table을 생성하는 함수

            Args:
                query: create query
        """
        self._set_db()

        try:
            with self.quant_db.cursor() as cursor:
                cursor.execute(query)
                self.quant_db.commit()
        finally:
            self._db_close()

    def set_commit(self, query, data, is_many=True):
        """
            data uploading하는 함수

            Args:
                query: str. uploading query
                data: list of list
                      ex) [['A005930', '2020-06-30', 50300.0],
                           ['A005930', '2020-07-01', 50309.0]]
                is_many: bool. 여러 row인지 여부
                        (default: True)

            query sample:
            #
                insert into
                    quantDB.multifactors (code, trade_date)
                values (%s, %s) as alias
                on duplicate key update
                    code = alias.code
                    trade_date = alias.trade_date
            #

        """
        self._set_db()

        try:
            with self.quant_db.cursor() as cursor:
                if is_many:
                    cursor.executemany(query, data)
                    self.quant_db.commit()
                else:
                    query = query %(dt for dt in data)
                    cursor.execute(query)
                    self.quant_db.commit()
        finally:
            self._db_close()

    def get_fetchall(self, query):
        """
            data 가져오는 함수

            Args:
                query: select query

            Return:
                result: selected data
        """
        self._set_db()

        try:
            with self.quant_db.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
        finally:
            self._db_close()

        return result

    @staticmethod
    def create_insert_query(table_name, cols):
        """
            DB 적재 쿼리를 생성하는 함수

            Args:
                table_name: str. 적재하려는 테이블 이름
                cols: list of str. table에서 적재하려는 column 명들
        """
        query = "insert into " + table_name + " ("

        if isinstance(cols, str):
            cols = [cols]

        col_n = len(cols)

        for i, col in enumerate(cols):
            if col_n != 1 and i < (col_n - 1):
                query += col + ","
            else:
                query += col + ")"

        query += " values ("

        for i in range(col_n):
            if col_n != 1 and i < (col_n - 1):
                query += "%s,"
            else:
                query += "%s) as alias \n"

        query += " on duplicate key update "

        for i, col in enumerate(cols):
            if col_n != 1 and i < (col_n - 1):
                query += col + " = alias." + col + ", "
            else:
                query += col + " = alias." + col + ";"

        return query
