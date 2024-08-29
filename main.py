# -*- coding = utf-8 -*-
# @Time: 2024/08/13
# @Author: Xinyue
# @File:main.py
# @Software: PyCharm

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import akshare as ak
import pandas_market_calendars as mcal
import statsmodels.api as sm
import matplotlib.pyplot as plt

from Utils.data_clean import DataProcess
from Utils.barra import Barra_Beta
from Utils.get_wind_data_per_stock import DataPerStock
from Utils.get_wind_data import WindData

start_time = '20081222'
end_time = datetime.now().strftime('%Y%m%d')
data = DataProcess(start_time, end_time)

csi_all_code = '000985.CSI'
csi_all = data.get_index_price(csi_all_code)
csi_all['TRADE_DT'] = pd.to_datetime(csi_all['TRADE_DT'])
csi_all[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']] = csi_all[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']].astype(float)
csi_all['CSI_RETURN'] = csi_all['S_DQ_CLOSE'] / csi_all['S_DQ_PRECLOSE'] - 1
print('中证全指(市值加权)：')
print(csi_all.head())

fields = 'S_INFO_WINDCODE,TRADE_DT,S_DQ_PRECLOSE,S_DQ_CLOSE,S_DQ_TRADESTATUS'
a_share = data.get_prices(fields)
a_share_nobj = a_share[~a_share['S_INFO_WINDCODE'].str.endswith('BJ')]
a_share_nobj.loc[:, 'TRADE_DT'] = pd.to_datetime(a_share_nobj['TRADE_DT'].values)

wind_a_ew_code = '8841388.WI'
my_ew_index_components = data.filter_index_cons(a_share_nobj, wind_a_ew_code)
my_ew_index_components[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']] = my_ew_index_components[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']].astype(float)
my_ew_index_components['EWI_RETURN'] = my_ew_index_components['S_DQ_CLOSE'] / my_ew_index_components['S_DQ_PRECLOSE'] - 1
my_ew_index = my_ew_index_components.groupby('TRADE_DT')['EWI_RETURN'].mean().reset_index()
print('万得全A剔除北交所：')
print(my_ew_index.head())

beta = Barra_Beta(start_time, end_time)
rf_df = beta.risk_free_rate()
rf_df.to_parquet('/Volumes/quanyi4g/factor/day_frequency/fundamental/RiskFree/risk_free.parquet')
fixed_df = beta.prepare_rm_n_rf(csi_all, my_ew_index)

# 立案调查事件
data = WindData(start_time, end_time)
reg_inv_df = data.get_reginv_info()
reg_inv_df['STR_ANNDATE'] = pd.to_datetime(reg_inv_df['STR_ANNDATE'])
reg_inv_df['STR_DATE'] = pd.to_datetime(reg_inv_df['STR_DATE'])
reg_inv_df['TRADE_DT'] = np.minimum(reg_inv_df['STR_DATE'], reg_inv_df['STR_ANNDATE'])
reg_inv_df = reg_inv_df[~reg_inv_df['S_INFO_WINDCODE'].str.endswith('BJ')]
stock_reg_inv = reg_inv_df['S_INFO_WINDCODE'].unique() # 找到发生过事件的标的
all_stock_reg_evt = pd.DataFrame()
length = len(stock_reg_inv)
count = 0
for stock_code in stock_reg_inv:
    beta_alpha_df = beta.beta_exposure(stock_code)
    all_stock_reg_evt = pd.concat([all_stock_reg_evt, beta_alpha_df])
    count += 1
    print(f'Finished {count}, stock code {stock_code}, {length - count} left')

all_stock_reg_evt = pd.merge(all_stock_reg_evt, reg_inv_df, how = 'left', on = ['TRADE_DT', 'S_INFO_WINDCODE'])
all_stock_reg_evt['if_reg_inv'] = all_stock_reg_evt[['SUR_REASONS', 'STR_ANNDATE', 'STR_DATE']].notna().any(axis=1).astype(int)
grouped = all_stock_reg_evt.groupby('S_INFO_WINDCODE')
reg_inv_ew_effect_df = pd.DataFrame()

for stock, group in grouped:
    group = group.reset_index(drop=True)
    event_indices = group.index[group['if_reg_inv'] == 1]
    if event_indices.empty:
        continue
    for i in range(len(event_indices)):
        start_idx = max(event_indices[i] - 30, 0)
        end_idx = min(event_indices[i] + 30, len(group) - 1)

        timerange_data = group.loc[start_idx:end_idx, ['EW_Alpha', 'CSI_Alpha']].reset_index(drop=True)
        timerange_data.columns = [f'EW_Alpha_{stock}_{i+1}', f'CSI_Alpha_{stock}_{i+1}']
        reg_inv_ew_effect_df = pd.concat([reg_inv_ew_effect_df, timerange_data], axis = 1)

ew_alpha_columns = reg_inv_ew_effect_df.filter(like='EW_Alpha_')
reg_inv_ew_effect_df['EW_Alpha_Avg'] = ew_alpha_columns.mean(axis=1)
reg_inv_ew_effect_df['Cumulative_Alpha'] = (1+reg_inv_ew_effect_df['EW_Alpha_Avg']).cumprod() - 1
time_index = range(-30, 31)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(time_index, reg_inv_ew_effect_df['EW_Alpha_Avg'], color='red', edgecolor='black', alpha=0.7, label='EW_Alpha Average')
ax1.set_xlabel('Time (Relative to Event)')
ax1.set_ylabel('EW_Alpha Average', color='black')
ax1.tick_params(axis='y', labelcolor='black')

ax2 = ax1.twinx()
ax2.plot(time_index, reg_inv_ew_effect_df['Cumulative_Alpha'], color='purple', label='Cumulative Alpha', linewidth=2)
ax2.set_ylabel('Cumulative Alpha', color='black')
ax2.tick_params(axis='y', labelcolor='black')

ax1.set_title('Regulation Investigation')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()

csi_alpha_columns = reg_inv_ew_effect_df.filter(like='CSI_Alpha_')
reg_inv_ew_effect_df['CSI_Alpha_Avg'] = csi_alpha_columns.mean(axis=1)
reg_inv_ew_effect_df['CSI_Cumulative_Alpha'] = (1+reg_inv_ew_effect_df['CSI_Alpha_Avg']).cumprod() - 1

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(time_index, reg_inv_ew_effect_df['CSI_Alpha_Avg'], color='red', edgecolor='black', alpha=0.7, label='CSI_Alpha Average')
ax1.set_xlabel('Time (Relative to Event)')
ax1.set_ylabel('CSI_Alpha Average', color='black')
ax1.tick_params(axis='y', labelcolor='black')

ax2 = ax1.twinx()
ax2.plot(time_index, reg_inv_ew_effect_df['CSI_Cumulative_Alpha'], color='purple', label='Cumulative CSI Alpha', linewidth=2)
ax2.set_ylabel('Cumulative CSI Alpha', color='black')
ax2.tick_params(axis='y', labelcolor='black')

ax1.set_title('Regulation Investigation')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()