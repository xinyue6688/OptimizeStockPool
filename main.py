# -*- coding = utf-8 -*-
# @Time: 2024/07/19
# @Author: Xinyue
# @File:main.py
# @Software: PyCharm

import pandas as pd
from datetime import datetime

from Utils.get_wind_data import WindData
from Utils.data_clean import DataProcess

start_time = '20100101'
end_time = datetime.now().strftime('%Y%m%d')
wind = WindData(start_time, end_time)

csi_all_code = '000985.CSI'
csi_all = wind.get_index_price(csi_all_code)
csi_all['TRADE_DT'] = pd.to_datetime(csi_all['TRADE_DT'])
csi_all[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']] = csi_all[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']].astype(float)
print(csi_all.head())

fields = ['S_INFO_WINDCODE','TRADE_DT','S_DQ_PRECLOSE','S_DQ_CLOSE','S_DQ_TRADESTATUS']
a_share = wind.get_prices(fields)
a_share_nobj = a_share[~a_share['S_INFO_WINDCODE'].str.endswith('BJ')]
print(a_share_nobj.head())

process = DataProcess(start_time, end_time)
wind_a_ew_code = '8841388.WI'
a_share_nobj['TRADE_DT'] = pd.to_datetime(a_share_nobj['TRADE_DT'].values)
my_ew_index = process.filter_index_cons(a_share_nobj, wind_a_ew_code)
print(my_ew_index.head())

rf = pd.read_excel('Data/riskfree.xlsx')
print(rf.head())


