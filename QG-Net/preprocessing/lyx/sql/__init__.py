# -*- coding: utf-8 -*-
import pymysql
from .. import common

def escape_string ( data ):
    return pymysql.escape_string(str(data))


def make_update_data ( data ):
    __update_data = [escape_string(v) for v in data[0].values()]
    __condition_datas = [escape_string(v) for v in data[1].values()]
    result = __update_data + __condition_datas
    return result



class DataBase():
    __db = None
    __cursor = None

    def __init__ ( self, database, host, user,
                   password, port = 3306, charset = 'utf8mb4' ):
        self.__db = pymysql.connect(host = host,
                                    port = port,
                                    user = user,
                                    password = password,
                                    database = database,
                                    charset = charset)
        self.__cursor = self.__db.cursor()

    def query ( self, sql_str ):
        '''
        查询
        :param sql_str:
        :return:
        '''
        try:
            self.__cursor.execute(sql_str)
            rows = self.__cursor.fetchall()
            return rows
        except:
            self.__db.rollback()
            print('query operation error')
            raise

    def executemany ( self, sql_str, data ):
        try:
            self.__cursor.executemany(sql_str, data)
            self.__db.commit()
        except:
            self.__db.rollback()
            print('Executemany operation error')
            raise

    def execute ( self, sql_str ):
        try:
            # print(sql_str)
            self.__cursor.execute(sql_str)
            self.__db.commit()
        except:
            self.__db.rollback()
            print('Execute many operation error')
            print(sql_str)
            raise

    def close_database ( self ):
        self.__cursor.close()
        self.__db.close()

    def table_clean ( self, tables ):
        if isinstance(tables, list):
            for table in tables:
                sql_str = 'truncate table %s' % table
                self.__cursor.execute(sql_str)
        else:
            sql_str = 'truncate table %s' % tables
            self.__cursor.execute(sql_str)

    def get_all_id ( self, table_name ):
        sql_str = 'SELECT id FROM %s' % table_name
        result = self.query(sql_str)
        result = [x[0] for x in result]
        return result

    def get_all ( self, fields, table_name, ids = None ):
        field_sql = ','.join(fields)
        if ids is None:
            # 不根据id取
            sql_str = 'SELECT {field_sql} FROM {table_name}'.format(
                    field_sql = field_sql, table_name = table_name)
            result = self.query(sql_str)
        else:
            # 根据id取
            result = []
            for id in ids:
                sql_str = 'SELECT {field_sql} FROM {table_name} where id ={id}'.format(
                        field_sql = field_sql, table_name = table_name, id = id)
                result.append(self.query(sql_str)[0])
        return result

    def insert_many ( self, mysql_table, data_list ):
        query = ""
        values = []

        for data_dict in data_list:
            if not query:
                columns = ', '.join('`{0}`'.format(k) for k in data_dict)
                place_holders = ', '.join('%s'.format(k) for k in data_dict)
                query = "INSERT IGNORE INTO {0} ({1}) VALUES ({2})".format(
                        mysql_table, columns, place_holders)

            v = [escape_string(x) for x in data_dict.values()]
            values.append(v)
        self.executemany(query, values)

    def update_many ( self, mysql_table,all_data ):
        prefix = 'update {mysql_table} set '.format(mysql_table = mysql_table)

        update_sql_list = []
        update_data=all_data[0][0]
        for field, data in update_data.items():
            sql_str = '{field}=%s '.format(field = field)
            update_sql_list.append(sql_str)
        update_sql = ', '.join(update_sql_list)

        condition_sql_list = []
        condition_data=all_data[0][1]
        for field, data in condition_data.items():
            sql_str = '{field}=%s '.format(field = field)
            condition_sql_list.append(sql_str)
        condition_sql = ', '.join(condition_sql_list)

        sql_str = prefix + update_sql + ' where ' + condition_sql

        result=list(common.mp_map(make_update_data,all_data))
        self.executemany(sql_str,result)
