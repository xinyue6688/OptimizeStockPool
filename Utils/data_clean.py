# -*- coding:UTF-8 -*-

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
from decimal import Decimal

from Utils.get_wind_data import WindData

class DataProcess(WindData):
    """
    数据处理类，用于筛选指数成分股，并分配行业信息。
    """

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

        super().__init__(self.start_date, self.end_date)
        self.get_data = super()

    def filter_index_cons(self, stock_list, index_code):
        """
        筛选指数成分股数据
        :param stock_list: 股票数据表 (pd.DataFrame)
        :param index_code: 指数代码 (str, 例:'000852.SH')
        :return: 指数成分股数据表 (pd.DataFrame)
        """
        index_member = self.get_data.get_index_con(index_code)
        index_member.drop(columns=['S_INFO_WINDCODE'], inplace=True)
        index_member.rename(columns={'S_CON_WINDCODE': 'S_INFO_WINDCODE'}, inplace=True)

        index_member['S_CON_INDATE'] = pd.to_datetime(index_member['S_CON_INDATE'].astype(str))
        index_member['S_CON_OUTDATE'] = pd.to_datetime(index_member['S_CON_OUTDATE'].astype(str),errors='coerce')

        start_date = sorted(index_member['S_CON_INDATE'].unique())[0]
        initiated_list = index_member[index_member['S_CON_INDATE'] == start_date]['S_INFO_WINDCODE']
        member_prices = pd.merge(stock_list, index_member, on='S_INFO_WINDCODE', how='left')
        member_prices = member_prices[
            ((member_prices['TRADE_DT'] >= member_prices['S_CON_INDATE']) &
            (member_prices['TRADE_DT'] <= member_prices['S_CON_OUTDATE'])) |
            (member_prices['S_CON_OUTDATE'].isna() & (member_prices['TRADE_DT'] >= member_prices['S_CON_INDATE'])) |
            ((member_prices['TRADE_DT'] < start_date) & (member_prices['S_INFO_WINDCODE'].isin(initiated_list)))
        ]
        before_date = member_prices[member_prices['TRADE_DT'] < start_date]
        before_date_unique = before_date.drop_duplicates(subset=['TRADE_DT', 'S_INFO_WINDCODE'])
        after_date = member_prices[member_prices['TRADE_DT'] >= start_date]
        member_prices = pd.concat([before_date_unique, after_date])
        member_prices.drop(columns=['S_CON_INDATE', 'S_CON_OUTDATE', 'CUR_SIGN'], inplace=True)
        return member_prices

    def assign_industry(self, stock_list):
        """
        分配行业信息
        :param stock_list: 股票数据表 (pd.DataFrame)
        :return: 分配行业信息后的股票数据表 (pd.DataFrame)
        """
        wind_primary_industry = pd.read_csv('Data/wind_primary_industry.csv',
                                            dtype={'INDUSTRIESCODE_PREFIX': str,
                                                   'INDUSTRIESCODE': str})
        ashareind_df = self.get_data.get_stock_ind()
        ashareind_df['REMOVE_DT'] = ashareind_df['REMOVE_DT'].astype(str).str.split('.').str[0]
        ashareind_df['REMOVE_DT'] = pd.to_datetime(ashareind_df['REMOVE_DT'], format='mixed',errors='coerce')
        ashareind_df['ENTRY_DT'] = pd.to_datetime(ashareind_df['ENTRY_DT'].astype(str))
        ashareind_df['WIND_IND_CODE'] = ashareind_df['WIND_IND_CODE'].astype(str)

        def map_primary_industry(industry_code, industry_classifier):
            """
            行业归类功能
            :param industry_code: 股票行业代码 (str, 例: '6220106020')
            :param industry_classifier: 行业分类标准 (pd.DataFrame)
            :return: 行业分类 (str)
            """
            for id, cls in zip(industry_classifier['INDUSTRIESCODE_PREFIX'], industry_classifier['WIND_NAME_ENG']):
                if industry_code.startswith(id):
                    return cls
            return None

        ashareind_df['WIND_PRI_IND'] = ashareind_df['WIND_IND_CODE'].apply(lambda x: map_primary_industry(x, wind_primary_industry))
        ashareind_df = ashareind_df[['S_INFO_WINDCODE', 'WIND_PRI_IND', 'ENTRY_DT', 'REMOVE_DT', 'CUR_SIGN']]
        stock_list_withind = pd.merge(stock_list, ashareind_df, on='S_INFO_WINDCODE', how='left')
        stock_list_withind = stock_list_withind[
            ((stock_list_withind['TRADE_DT'] >= stock_list_withind['ENTRY_DT']) &
             ((stock_list_withind['TRADE_DT'] <= stock_list_withind['REMOVE_DT']) | stock_list_withind['REMOVE_DT'].isna())) |
            (stock_list_withind['REMOVE_DT'].isna() & (stock_list_withind['TRADE_DT'] >= stock_list_withind['ENTRY_DT']))
        ]
        stock_list_withind.drop(columns=['ENTRY_DT', 'REMOVE_DT', 'CUR_SIGN'], inplace=True)
        return stock_list_withind

    def add_future_rt(self, stock_list):
        """
        添加未来收益列
        :param stock_list: 包含当日收盘价、昨日收盘价、和交易日的数据表 (pd.DataFrame)
        :return: 计算未来收益后的数据表 (pd.DataFrame)
        """
        stock_list['RETURN'] = stock_list['S_DQ_ADJCLOSE']/stock_list['S_DQ_ADJPRECLOSE'] - 1
        stock_list['RETURN_NXT'] = stock_list.groupby('S_INFO_WINDCODE')['RETURN'].shift(-1)
        stock_list.dropna(subset=['RETURN_NXT'], inplace=True)
        return stock_list

    def mv_neutralize(self, df):
        """
        市值中性化处理
        :param df: 包含原始市值和标准化因子值的数据表 (pd.DataFrame)
        :return: 市值中性化处理后的数据表 (pd.DataFrame)
        """
        df['const'] = 1.0

        # Convert Decimal to float if necessary
        df['S_VAL_MV'] = df['S_VAL_MV'].apply(lambda x: float(x) if isinstance(x, Decimal) else x)

        df['lnMV'] = np.log(df['S_VAL_MV'])
        df.dropna(inplace=True)

        X = df[['const', 'lnMV']]
        y = df['S_DQ_TURN_norm']

        # Ensure the data is numeric and has no NaNs
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')

        X.dropna(inplace=True)
        y = y[X.index]

        if X.isnull().values.any() or y.isnull().values.any():
            raise ValueError("Data contains NaN values after conversion to numeric types")

        model = sm.OLS(y, X)
        results = model.fit()
        df['turnover'] = results.resid
        df.drop(['S_DQ_TURN_norm', 'const'], axis=1, inplace=True)

        return df

if __name__ == "__main__":
    start = '20100101'
    end = '20100105'
    fields = ['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_ADJPRECLOSE', 'S_DQ_ADJCLOSE', 'S_DQ_TRADESTATUS']

    data_processer = DataProcess(start, end)
    data = data_processer.get_prices(fields)

    data['TRADE_DT'] = pd.to_datetime(data['TRADE_DT'].astype('str'))
    data_csi = data_processer.filter_index_cons(data, '000852.SH')
    print(data_csi.head())

