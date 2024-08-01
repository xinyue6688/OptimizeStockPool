# -*- coding = utf-8 -*-

from Utils.connect_wind import ConnectDatabase


class WindData(ConnectDatabase):
    '''获取wind数据'''
    def __init__(self, start_date: str, end_date: str):
        '''
        :param start_date: 起始时间 例：‘20100101’
        :param end_date: 结束时间 例：datetime.datetime.now().strftime('%Y%m%d')
        '''
        self.start_date = start_date
        self.end_date = end_date

    def get_prices(self, fields):
        '''获取行情数据'''
        fields_sql = ', '.join(fields)
        table = 'ASHAREEODPRICES'

        sql = f'''SELECT {fields_sql}               
                FROM {table}
                WHERE (TRADE_DT BETWEEN '{self.start_date}' AND '{self.end_date}')
            '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='TRADE_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_index_con(self, index_code: str):
        '''获取指数成分股'''
        table = 'AINDEXMEMBERS'
        fields = ['S_INFO_WINDCODE', 'S_CON_WINDCODE', 'S_CON_INDATE', 'S_CON_OUTDATE', 'CUR_SIGN']
        fields_sql = ', '.join(fields)

        sql = f'''SELECT {fields_sql}               
                        FROM {table}
                        WHERE (S_INFO_WINDCODE = '{index_code}')
                    '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='S_CON_INDATE', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_stock_ind(self):
        '''获取股票行业分类'''
        table = 'ASHAREINDUSTRIESCLASS'
        fields = ['S_INFO_WINDCODE','WIND_CODE','WIND_IND_CODE','ENTRY_DT','REMOVE_DT','CUR_SIGN']
        fields_sql = ', '.join(fields)

        sql = f'''SELECT {fields_sql}              
                  FROM {table}
                '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='ENTRY_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_indicator(self, fields):
        '''获取行情衍生数据'''
        fields_sql = ', '.join(fields)
        table = 'ASHAREEODDERIVATIVEINDICATOR'

        sql = f'''SELECT {fields_sql}               
                        FROM {table}
                        WHERE (TRADE_DT BETWEEN '{self.start_date}' AND '{self.end_date}')
                    '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='TRADE_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_industry_index(self, index_code: str):
        fields = ['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_PRECLOSE', 'S_DQ_CLOSE', 'S_DQ_PCTCHANGE']
        table = 'AINDEXWINDINDUSTRIESEOD'

        fields_sql = ', '.join(fields)

        sql = f'''SELECT {fields_sql}               
                     FROM {table}
                     WHERE (TRADE_DT BETWEEN '{self.start_date}' AND '{self.end_date}')
                     AND (S_INFO_WINDCODE = '{index_code}')
                  '''
        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='TRADE_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_all_industries(self):
        fields = ['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_PRECLOSE', 'S_DQ_CLOSE']
        fields_sql = ', '.join(fields)
        table = 'AINDEXWINDINDUSTRIESEOD'

        sql = f'''SELECT {fields_sql}               
                     FROM {table}
                     WHERE (TRADE_DT BETWEEN '{self.start_date}' AND '{self.end_date}')
                     AND (S_INFO_WINDCODE LIKE '8820__.WI' OR S_INFO_WINDCODE = '8841388.WI')
                  '''
        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='TRADE_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


if __name__ == '__main__':
    start_time = '20100101'
    end_time = '20100105'
    wind_data = WindData(start_time, end_time)

    fields = ['S_INFO_WINDCODE','TRADE_DT','S_DQ_ADJPRECLOSE','S_DQ_ADJCLOSE','S_DQ_TRADESTATUS']
    prices = wind_data.get_prices(fields)
    print(prices.head())

    index_code = '000852.SH'
    csi1000_cons = wind_data.get_index_con(index_code)
    print(csi1000_cons.head())

    industry_assign = wind_data.get_stock_ind()
    print(industry_assign.head())

    dev_fields = ['S_INFO_WINDCODE','TRADE_DT','S_VAL_MV','S_DQ_TURN']
    indicators = wind_data.get_indicator(dev_fields)
    print(indicators.head())

    index_code_2 = '8841388.WI'
    wind_a_index = wind_data.get_industry_index(index_code_2)
    print(wind_a_index.head())

    all_ind = wind_data.get_all_industries()
    print(all_ind.head())
