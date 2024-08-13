# -*- coding = utf-8 -*-
# @Time: 2024/07/19
# @Author: Xinyue
# @File:main.py
# @Software: PyCharm

import pandas as pd
from datetime import datetime
import numpy as np
import pandas_market_calendars as mcal
import statsmodels.api as sm
import matplotlib.pyplot as plt

from Utils.connect_wind import ConnectDatabase
from Utils.get_wind_data import WindData
from Utils.data_clean import DataProcess

start_time = '20081222'
end_time = datetime.now().strftime('%Y%m%d')
data = DataProcess(start_time, end_time)

csi_all_code = '000985.CSI'
csi_all = data.get_index_price(csi_all_code)
csi_all['TRADE_DT'] = pd.to_datetime(csi_all['TRADE_DT'])
csi_all[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']] = csi_all[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']].astype(float)
csi_all['CSI_RETURN'] = csi_all['S_DQ_CLOSE'] / csi_all['S_DQ_PRECLOSE'] - 1
print(csi_all.head())

fields = 'S_INFO_WINDCODE,TRADE_DT,S_DQ_PRECLOSE,S_DQ_CLOSE,S_DQ_TRADESTATUS'
a_share = data.get_prices(fields)
a_share_nobj = a_share[~a_share['S_INFO_WINDCODE'].str.endswith('BJ')]
a_share_nobj.loc[:, 'TRADE_DT'] = pd.to_datetime(a_share_nobj['TRADE_DT'].values)
print(a_share_nobj.head())

wind_a_ew_code = '8841388.WI'
my_ew_index_components = data.filter_index_cons(a_share_nobj, wind_a_ew_code)
my_ew_index_components[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']] = my_ew_index_components[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']].astype(float)
my_ew_index_components['EWI_RETURN'] = my_ew_index_components['S_DQ_CLOSE'] / my_ew_index_components['S_DQ_PRECLOSE'] - 1
my_ew_index = my_ew_index_components.groupby('TRADE_DT')['EWI_RETURN'].mean().reset_index()

market_returns = pd.merge(csi_all[['TRADE_DT', 'CSI_RETURN']], my_ew_index[['TRADE_DT', 'EWI_RETURN']], on='TRADE_DT')
print(market_returns.head())
market_returns = market_returns[['TRADE_DT', 'EWI_RETURN', 'CSI_RETURN']]
#market_returns.to_parquet('Data/beta.parquet')

rf = pd.read_excel('Data/riskfree.xlsx')
rf.rename(columns={
    '指标名称': 'TRADE_DT',
    '国债到期收益率:10年': 'RF_RETURN'
}, inplace=True)
rf = rf.iloc[:-2]
rf['RF_RETURN'] = rf['RF_RETURN']/100
#rf.to_parquet('Data/riskfree.parquet')
rf['TRADE_DT'] = pd.to_datetime(rf['TRADE_DT'])
fixed_df = pd.merge(rf, market_returns, on='TRADE_DT', how = 'left') # 格式：日期、市值加权beta、等权beta

# 观测beta暴露的窗口期为252天，半衰期为63天
window_size = 252
half_life = 63
lambda_ = np.exp(-np.log(2)/half_life)
t = np.arange(1, window_size + 1)
weights = lambda_ ** t
weights = weights[::-1]

# 获取中国市场的交易日历
cn_cal = mcal.get_calendar('SSE')  # 'SSE'代表上海证券交易所

tradable_components = pd.read_parquet('Data/tradable_components.parquet')
grouped = tradable_components.groupby('S_INFO_WINDCODE')
length = len(grouped)
count = 0
problem_list = []
group_names = list(grouped.groups.keys())

for name in group_names:
    group = grouped.get_group(name)

    first_day = group['TRADE_DT'].iloc[0]  # 成分股第一次加入池子的时间
    last_day = group['TRADE_DT'].iloc[-1]  # 成分股最后一次出现在池子的时间
    schedule = cn_cal.schedule(start_date='2008-12-22', end_date=first_day)
    trading_days = schedule.index
    data_start_date = trading_days[trading_days < first_day][-window_size].strftime('%Y%m%d')
    data_last_date = last_day.strftime('%Y%m%d')

    # 每只成分股的数据从第一次加入池子的前252天取到最后一天
    stock_data = DataProcess(data_start_date, data_last_date).get_sngstock_price('TRADE_DT, S_DQ_PRECLOSE, S_DQ_CLOSE', name)
    stock_data[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']] = stock_data[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']].astype(float)
    stock_data['STOCK_RETURN'] = stock_data['S_DQ_CLOSE'] / stock_data['S_DQ_PRECLOSE'] - 1
    stock_data['TRADE_DT'] = pd.to_datetime(stock_data['TRADE_DT'])
    stock_data = stock_data.merge(fixed_df, on='TRADE_DT', how='left')

    # 计算成分股市场暴露
    beta_results = []
    for i in range(len(group['TRADE_DT'])):
        window_size = 252
        t = group['TRADE_DT'].iloc[i]
        index = stock_data[stock_data['TRADE_DT'] == t].index[0]
        window_data = stock_data.iloc[index - np.minimum(index, window_size): index] # 如果index小于等于252，回归用到的期数=index；如果index大于252，回归用到的期数为252
        window_data = window_data.dropna()

        # 准备回归数据
        reg_data = window_data[['STOCK_RETURN', 'RF_RETURN', 'EWI_RETURN', 'CSI_RETURN']].copy()
        reg_data['y'] = reg_data['STOCK_RETURN'] - reg_data['RF_RETURN']
        reg_data['const'] = 1  # Add a constant term

        if len(reg_data) != len(weights):
            weights_adj = weights[-len(reg_data):]
        else:
            weights_adj = weights

        try:
            EW_model = sm.WLS(reg_data['y'], reg_data[['const', 'EWI_RETURN']], weights=weights_adj).fit()
            EW_beta_t = EW_model.params['EWI_RETURN']
        except Exception as e:
            problem_list.append(name)
            print(f"EW_model error: {e} at iteration {i}")
            print(reg_data)
            continue

        try:
            CSI_model = sm.WLS(reg_data['y'], reg_data[['const', 'CSI_RETURN']], weights=weights_adj).fit()
            CSI_beta_t = CSI_model.params['CSI_RETURN']
        except Exception as e:
            problem_list.append(name)
            print(f"CSI_model error: {e} at iteration {i}")
            print(reg_data)
            continue

        beta_results.append({
            'TRADE_DT': t,
            'EW_Beta': EW_beta_t,
            'CSI_Beta': CSI_beta_t
        })
    # 如果股票在 problem_list 中，则跳过后续操作
    if name in problem_list:
        print(f'Skipping {name} due to previous errors.')
        continue

    beta_df = pd.DataFrame(beta_results)
    beta_df['TRADE_DT'] = pd.to_datetime(beta_df['TRADE_DT'])

    try:
        beta_df = beta_df.merge(stock_data, on='TRADE_DT', how='left')
    except KeyError as e:
        print(f"Merge error for stock {name}: {e}")
        print("beta_df columns:", beta_df.dtypes)
        print("stock_data columns:", stock_data.dtypes)
        continue

    beta_df['EW_Alpha'] = beta_df['STOCK_RETURN'] - beta_df['RF_RETURN'] - beta_df['EWI_RETURN'] * beta_df['EW_Beta']
    beta_df['CSI_Alpha'] = beta_df['STOCK_RETURN'] - beta_df['RF_RETURN'] - beta_df['CSI_RETURN'] * beta_df['CSI_Beta']
    beta_df = beta_df[['TRADE_DT', 'EW_Beta', 'CSI_Beta', 'EW_Alpha', 'CSI_Alpha']]
    beta_df.to_parquet(f'Component_beta/{name.replace(".", "_")}_beta.parquet')

    count += 1
    print(f'{name} DONE, COMPLETED WRITING {count}, {length - count} LEFT')

for stock in problem_list:
    print(stock)