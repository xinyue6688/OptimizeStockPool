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
from Utils.get_wind_data_per_stock import DataPerStock

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

market_returns = pd.merge(csi_all[['TRADE_DT', 'CSI_RETURN']], my_ew_index[['TRADE_DT', 'EWI_RETURN']], on='TRADE_DT')

print(market_returns.head())
market_returns = market_returns[['TRADE_DT', 'EWI_RETURN', 'CSI_RETURN']]
#market_returns.to_parquet('Data/beta.parquet')

current_start_time = datetime.strptime(start_time, "%Y%m%d")

bond_china_yield_combined_df = pd.DataFrame()

while current_start_time.strftime("%Y%m%d") < end_time:
    current_end_time = min(current_start_time + timedelta(days=365), datetime.strptime(end_time, "%Y%m%d"))

    bond_china_yield_df = ak.bond_china_yield(
        start_date=current_start_time.strftime("%Y%m%d"),
        end_date=current_end_time.strftime("%Y%m%d")
    )

    bond_china_yield_filtered_df = bond_china_yield_df[
        (bond_china_yield_df['曲线名称'] == '中债国债收益率曲线')
    ][['日期', '10年']]

    bond_china_yield_combined_df = pd.concat([bond_china_yield_combined_df, bond_china_yield_filtered_df])

    current_start_time = current_end_time + timedelta(days=1)

bond_china_yield_combined_df.reset_index(drop=True, inplace=True)
rf = bond_china_yield_combined_df.rename(columns={'日期': 'TRADE_DT',
                                                  '10年': 'RF_RETURN_ANN'})
rf['RF_RETURN_ANN'] = rf['RF_RETURN_ANN']/100
rf.to_excel('Data/risk_free.xlsx')
rf['RF_RETURN'] = (1 + rf['RF_RETURN_ANN']) ** (1/252) - 1
rf['TRADE_DT'] = pd.to_datetime(rf['TRADE_DT'])
rf.drop(columns=['RF_RETURN_ANN'], inplace=True)
fixed_df = pd.merge(rf, market_returns, on='TRADE_DT', how = 'left') # 格式：日期、市值加权beta、等权beta

# 观测beta暴露的窗口期为252天，半衰期为63天
window_size = 252
half_life = 63
lambda_ = np.exp(-np.log(2) / half_life)
t = np.arange(1, window_size + 1)
weights = lambda_ ** t
weights = weights[::-1]

# 获取中国市场的交易日历
cn_cal = mcal.get_calendar('SSE')  # 'SSE'代表上海证券交易所

grouped = a_share_nobj.groupby('S_INFO_WINDCODE')
length = len(grouped)
count = 0
problem_list = []
pool_alpha_df = pd.DataFrame()

for name, group in grouped:
    # 每只成分股的数据从2010/01/04的前252个交易日（2008/12/22）取到今天
    # 可能会出现数据长度不满252天的情况
    stock_data = group.copy()
    if len(stock_data) < 10:
        print(f'Stock {name} is listed for less than 10 days, not enough data to calculate market exposure')
        problem_list.append(name)
        count += 1
        continue

    stock_data[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']] = stock_data[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']].astype(float)
    stock_data['STOCK_RETURN'] = stock_data['S_DQ_CLOSE'] / stock_data['S_DQ_PRECLOSE'] - 1
    stock_data['TRADE_DT'] = pd.to_datetime(stock_data['TRADE_DT'])
    stock_data = stock_data.merge(fixed_df, on='TRADE_DT', how='left')

    # 计算成分股市场暴露
    beta_results = []
    matching_index = stock_data[stock_data['TRADE_DT'] == '2010-01-04'].index
    if not matching_index.empty:
        start_index = matching_index[0]
    else:
        start_index = 10
    for index in range(start_index, len(group['TRADE_DT'])):
        window_data = stock_data.iloc[max(0, index - window_size): index].dropna()
        t = stock_data.loc[index, 'TRADE_DT']
        # 准备回归数据
        reg_data = window_data[['S_INFO_WINDCODE', 'STOCK_RETURN', 'RF_RETURN', 'EWI_RETURN', 'CSI_RETURN']].copy()
        reg_data['y'] = reg_data['STOCK_RETURN'] - reg_data['RF_RETURN']
        reg_data['const'] = 1  # Add a constant term

        weights_adj = weights[-len(reg_data):] if len(reg_data) != len(weights) else weights

        try:
            EW_model = sm.WLS(reg_data['y'], reg_data[['const', 'EWI_RETURN']], weights=weights_adj).fit()
            EW_beta_t = EW_model.params['EWI_RETURN']
        except Exception as e:
            problem_list.append(name)
            print(f"EW_model error: {e} at iteration {index}")
            continue

        try:
            CSI_model = sm.WLS(reg_data['y'], reg_data[['const', 'CSI_RETURN']], weights=weights_adj).fit()
            CSI_beta_t = CSI_model.params['CSI_RETURN']
        except Exception as e:
            problem_list.append(name)
            print(f"CSI_model error: {e} at iteration {index}")
            continue

        beta_results.append({
            'TRADE_DT': t,
            'EW_Beta': EW_beta_t,
            'CSI_Beta': CSI_beta_t
        })

    if name in problem_list:
        print(f'Skipping {name} due to previous errors.')
        continue

    beta_df = pd.DataFrame(beta_results)
    beta_df['TRADE_DT'] = pd.to_datetime(beta_df['TRADE_DT'])

    try:
        beta_df = beta_df.merge(stock_data, on='TRADE_DT', how='left')
    except KeyError as e:
        print(f"Merge error for stock {name}: {e}")
        continue

    beta_df['EW_Alpha'] = beta_df['STOCK_RETURN'] - beta_df['RF_RETURN'] - beta_df['EWI_RETURN'] * beta_df['EW_Beta']
    beta_df['CSI_Alpha'] = beta_df['STOCK_RETURN'] - beta_df['RF_RETURN'] - beta_df['CSI_RETURN'] * beta_df['CSI_Beta']
    beta_df = beta_df[['TRADE_DT', 'EW_Beta', 'CSI_Beta', 'EW_Alpha', 'CSI_Alpha']]
    beta_df.to_parquet(f'Component_beta/{name.replace(".", "_")}_beta.parquet')

    beta_df['S_INFO_WINDCODE'] = name
    pool_alpha_df = pd.concat([pool_alpha_df, beta_df]).reset_index(drop=True)
    count += 1
    print(f'{name} DONE, COMPLETED WRITING {count}, {length - count} LEFT')

print(pool_alpha_df.head())


