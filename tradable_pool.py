# -*- coding = utf-8 -*-
# @Time: 2024/07/31
# @Author: Xinyue
# @File:tradable_pool.py
# @Software: PyCharm

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from scipy.stats import zscore
from decimal import Decimal
import seaborn as sns

from Utils.data_clean import DataProcess
from Utils.return_metrics import MetricsCalculator
from Utils.factor_test import FactorDecileAnalysis

start_time = '20100101'
end_time = datetime.now().strftime('%Y%m%d')
data = DataProcess(start_time, end_time)
# 获取全市场行情
price = data.get_prices('S_INFO_WINDCODE,TRADE_DT,S_DQ_PRECLOSE,S_DQ_OPEN,S_DQ_CLOSE,S_DQ_TRADESTATUS,S_DQ_LIMIT,S_DQ_STOPPING, S_DQ_AMOUNT')
print(price.head())
price['TRADE_DT'] = pd.to_datetime(price['TRADE_DT'])
price[['S_DQ_PRECLOSE','S_DQ_OPEN','S_DQ_CLOSE','S_DQ_LIMIT','S_DQ_STOPPING']] = price[['S_DQ_PRECLOSE','S_DQ_OPEN','S_DQ_CLOSE','S_DQ_LIMIT','S_DQ_STOPPING']].astype(float)
price.sort_values(by='TRADE_DT', ascending = True, inplace = True)
print(price.head())
print('-----PRICE DATA DOWNLOADED-----')
# 跟踪WIND全A指数8841388.WI
price_winda = data.filter_index_cons(price, '8841388.WI')
print('-----FOLLOWED WIND A INDEX------')
# 剔除北交所
price_exbj = price_winda[~price_winda['S_INFO_WINDCODE'].str.endswith('BJ')]
print('-----EXCLUDED BJ STOCKS------')
# Shift未来收益
price_exbj['RETURN'] = price_exbj['S_DQ_CLOSE'] / price_exbj['S_DQ_PRECLOSE'] - 1
price_exbj['RETURN_NXT'] = price_exbj.groupby(('S_INFO_WINDCODE'))['RETURN'].shift(-1)
# 市场组合收益
all_market_rt = price_exbj.groupby('TRADE_DT')['RETURN_NXT'].mean().reset_index()
print('ALL MARKET AVERAGE RETURN:')
print(all_market_rt.head())

'''标记剔除条件部分'''

#### 次新股 ####
list_info = ['S_INFO_WINDCODE',
             'S_INFO_LISTDATE',
             'S_INFO_DELISTDATE']
list_data = data.get_stock_dsp(list_info)
list_data['S_INFO_LISTDATE'] = pd.to_datetime(list_data['S_INFO_LISTDATE'].str.split('.').str[0])
list_data['S_INFO_DELISTDATE'] = pd.to_datetime(list_data['S_INFO_DELISTDATE'].str.split('.').str[0])
price_listinfo = pd.merge(price_exbj, list_data, on='S_INFO_WINDCODE', how='left')

def DetermineNewStatus(row):
    if pd.notna(row['S_INFO_DELISTDATE']) and row['TRADE_DT'] >= row['S_INFO_DELISTDATE']:
        return 2 # Delisted stock
    elif row['TRADE_DT'] < row['S_INFO_LISTDATE'] + pd.DateOffset(years=1):
        return 1 # Subnew stock
    else:
        return 0 # Normal stock

price_listinfo['SUBNEW'] = price_listinfo.apply(DetermineNewStatus, axis=1)

filter_relist = price_listinfo[(price_listinfo['S_DQ_TRADESTATUS'] == 'N') & (price_listinfo['SUBNEW'] == 0)] # 修正重新上市的股票
for name in filter_relist['S_INFO_WINDCODE'].unique():
    new_trade_date = filter_relist[filter_relist['S_INFO_WINDCODE'] == name]['TRADE_DT'].values[0]
    price_listinfo.loc[(price_listinfo['S_INFO_WINDCODE'] == name)& (price_listinfo['TRADE_DT'] >= new_trade_date), 'S_INFO_LISTDATE'] = new_trade_date

price_listinfo['SUBNEW'] = price_listinfo.apply(DetermineNewStatus, axis=1)
price_subnew_revised = price_listinfo.drop(columns=['S_INFO_LISTDATE', 'S_INFO_DELISTDATE'])
print('-----SUBNEW MARKED-----')

#### 停牌 ####
sus_df = data.get_suspend_info('S_INFO_WINDCODE, S_DQ_SUSPENDDATE, S_DQ_RESUMPDATE')
sus_df['S_DQ_SUSPENDDATE'] = pd.to_datetime(sus_df['S_DQ_SUSPENDDATE'])
price_suspend = pd.merge(price_subnew_revised, sus_df, left_on = ['S_INFO_WINDCODE', 'TRADE_DT'], right_on = ['S_INFO_WINDCODE', 'S_DQ_SUSPENDDATE'], how='left')
price_suspend['SUSPEND'] = 0
mask_filter_suspend = price_suspend['S_DQ_SUSPENDDATE'].notna()
price_suspend.loc[mask_filter_suspend, 'SUSPEND'] = 1
price_suspend['PAST_10_SUSPEND'] = price_suspend.groupby('S_INFO_WINDCODE')['SUSPEND'].rolling(window=10, min_periods=1).sum().reset_index(0, drop=True)
price_suspend['SUSPEND'] = (price_suspend['PAST_10_SUSPEND'] > 0).astype(int)
price_suspend = price_suspend.drop(columns=['PAST_10_SUSPEND', 'S_DQ_SUSPENDDATE', 'S_DQ_RESUMPDATE'])
print('-----SUSPEND MARKED-----')

#### ST ####
st_data = data.get_st_info()
st_data['ENTRY_DT'] = pd.to_datetime(st_data['ENTRY_DT'])
st_data['REMOVE_DT'] = pd.to_datetime(st_data['REMOVE_DT'].str.split('.').str[0])
st_data['ANN_DT'] = pd.to_datetime(st_data['ANN_DT'])
st_data.loc[st_data['ANN_DT'] < st_data['ENTRY_DT'], 'ENTRY_DT'] = st_data['ANN_DT']

expanded_rows = []

for index, row in st_data.iterrows():
    current_date = row['ENTRY_DT']
    end_date = row['REMOVE_DT'] if pd.notna(row['REMOVE_DT']) else pd.Timestamp('today')
    while current_date <= end_date:
        expanded_rows.append({
            'S_INFO_WINDCODE': row['S_INFO_WINDCODE'],
            'TRADE_DT': current_date,
            'S_TYPE_ST': row['S_TYPE_ST']
        })
        current_date += pd.Timedelta(days=1)

expanded_ststatus = pd.DataFrame(expanded_rows)
price_st = pd.merge(price_suspend, expanded_ststatus, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='left')
price_st['ST_STATUS'] = price_st.groupby(['S_INFO_WINDCODE', 'TRADE_DT'])['S_TYPE_ST'].transform(lambda x: ','.join(x.dropna().unique()))
price_st = price_st[['S_INFO_WINDCODE', 'TRADE_DT', 'ST_STATUS']].drop_duplicates().reset_index(drop=True)
price_st['ST_STATUS'] = price_st['ST_STATUS'].replace('', 'Normal')
price_st = pd.merge(price_suspend, price_st, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='left')
print('-----ST MARKED-----')

#### 净资产、流动性、市值 ####
indicator_info = 'S_INFO_WINDCODE,TRADE_DT,NET_ASSETS_TODAY,S_DQ_TURN,S_VAL_MV'
indicator_data = data.get_indicator(indicator_info)
indicator_data['TRADE_DT'] = pd.to_datetime(indicator_data['TRADE_DT'])
price_indicators = pd.merge(price_st, indicator_data, on=['S_INFO_WINDCODE','TRADE_DT'], how='left')
#rows_with_na = price_indicators[price_indicators[['NET_ASSETS_TODAY', 'S_DQ_TURN', 'S_VAL_MV', 'S_DQ_AMOUNT']].isna().any(axis=1)]
#print(rows_with_na)
group_by_code = price_indicators.groupby('S_INFO_WINDCODE') # 向后填充
def fillna_forward(group):
    group['NET_ASSETS_TODAY'] = group['NET_ASSETS_TODAY'].fillna(method='ffill')
    return group

price_indicators = group_by_code.apply(fillna_forward).reset_index(drop = True)

asset_floor = 0
liquid_floor = 1000000 / 1000  # 单位为千元 (100万元)
mv_floor = 300000000 / 10000    # 单位为万元 (3亿元)
price_indicators['NEG_ASSET'] = 0
price_indicators['MINI_MV'] = 0
price_indicators['LOW_LIQUIDITY'] = 0

price_indicators.loc[price_indicators['NET_ASSETS_TODAY'] < asset_floor, 'NEG_ASSET'] = 1 # 标记净资产为负
print('-----NEGATIVE ASSET MARKED-----')
price_indicators.loc[price_indicators['S_VAL_MV'] < mv_floor, 'MINI_MV'] = 1 # 标记极小市值
print('-----EXTREME SMALL MV MARKED-----')
price_indicators['S_DQ_AMOUNT'] = price_indicators['S_DQ_AMOUNT'].astype(float)
quantiles_10 = price_indicators.groupby('TRADE_DT')['S_DQ_AMOUNT'].transform(lambda x: x.quantile(0.10))
price_indicators.loc[(price_indicators['S_DQ_AMOUNT'] <= quantiles_10) | (price_indicators['S_DQ_AMOUNT'] <= liquid_floor), 'LOW_LIQUIDITY'] = 1 # 历史截面小于10%
price_indicators['LOW_LIQUIDITY'] = price_indicators.groupby('S_INFO_WINDCODE')['LOW_LIQUIDITY'].shift(1).fillna(0)
print('-----LOW LIQUIDITY MARKED-----')
print('FILTER CONDITIONS ALL MARKED')

# Filter stocks that qualifies the exclusion requirements
data_all = price_indicators.copy()

problem_list = ['000403.SZ', '000620.SZ', '000981.SZ'] # 借壳上市股票
for stock in problem_list:
    idx = data_all[data_all['S_INFO_WINDCODE'] == stock].index[:252]
    data_all.loc[idx, 'SUBNEW'] = 1

# Filter stocks that qualifies the exclusion requirements
exclude_mask = (data_all['SUBNEW'] != 0) |\
               (data_all['SUSPEND'] == 1) |\
               (data_all['ST_STATUS'] != 'Normal') |\
               (data_all['NEG_ASSET'] == 1) |\
               (data_all['MINI_MV'] == 1) |\
               (data_all['LOW_LIQUIDITY'] == 1)

my_pool = data_all[~exclude_mask]
my_pool.sort_values(by = ['TRADE_DT'], ascending = True, inplace = True)
tradable_components = my_pool[['TRADE_DT', 'S_INFO_WINDCODE']].reset_index(drop = True)
tradable_components.to_parquet('Data/tradable_components.parquet', engine = 'pyarrow', index = False)

conditions = ['SUBNEW', 'SUSPEND', 'ST_STATUS', 'NEG_ASSET', 'MINI_MV', 'LOW_LIQUIDITY']
results = {'TRADE_DT': data_all['TRADE_DT'].unique().to_numpy()}
grouped_by_day = data_all.groupby('TRADE_DT')

for condition in conditions:
    results[condition] = np.zeros(len(results['TRADE_DT']))

    if condition == 'ST_STATUS':
        daily_num = grouped_by_day.apply(lambda x: (x[condition] != 'Normal').sum())
    elif condition == 'SUBNEW':
        daily_num = grouped_by_day.apply(lambda x: (x[condition] != 0).sum())
    else:
        daily_num = grouped_by_day.apply(lambda x: (x[condition] == 1).sum())

    results[condition] = daily_num

results['ALL MARKET'] = data_all.groupby(['TRADE_DT'])['S_INFO_WINDCODE'].nunique().values
results['STOCKS REMAINING'] = my_pool.groupby(['TRADE_DT'])['S_INFO_WINDCODE'].nunique().values

daily_filter_stats = pd.DataFrame(results).reset_index(drop=True)
daily_filter_stats.to_excel('Data/逐日股票池剔除数量表(修正版).xlsx', index=False)

my_pool_rt = my_pool.groupby('TRADE_DT')['RETURN_NXT'].mean().reset_index()
my_pool_rt.rename(columns={'RETURN_NXT': 'RETURN_MYPOOL'}, inplace=True)
print('----FILTERED POOL----')
print(my_pool_rt.head())

# 刚性优化后的池子和全市场对比
pool_vs_allmarket = pd.merge(my_pool_rt, all_market_rt, on=['TRADE_DT'], how='left')

pool_vs_allmarket['NAV_MYPOOL'] = (pool_vs_allmarket['RETURN_MYPOOL'] + 1).cumprod()
pool_vs_allmarket['NAV_MARKET'] = (pool_vs_allmarket['RETURN_NXT'] + 1).cumprod()
pool_vs_allmarket['NSTOCK_MYPOOL'] = my_pool.groupby('TRADE_DT')['S_INFO_WINDCODE'].nunique().reset_index(drop=True)
pool_vs_allmarket['NSTOCK_MARKET'] = data_all.groupby('TRADE_DT')['S_INFO_WINDCODE'].nunique().reset_index(drop=True)

plt.plot(pool_vs_allmarket['TRADE_DT'], pool_vs_allmarket['NAV_MYPOOL'], label='My Pool')
plt.plot(pool_vs_allmarket['TRADE_DT'], pool_vs_allmarket['NAV_MARKET'], label='All Market')
plt.title('NAV Comparison')
plt.xlabel('Trade Date')
plt.ylabel('NAV')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 年化收益和收益率特性
def annualized_return(returns_series):
    num_days = len(returns_series)
    total_return = np.prod(1+returns_series)
    annualized_return = total_return ** (252 / num_days) - 1
    return annualized_return

pool_vs_allmarket.dropna(inplace=True)
pool_vs_allmarket['RETURN_DIFF'] = pool_vs_allmarket['RETURN_MYPOOL'] - pool_vs_allmarket['RETURN_NXT']

annualized_returns_compare = {}

annualized_returns_compare[f'Filtering Effect'] = annualized_return(pool_vs_allmarket[f'RETURN_DIFF'])
annualized_returns_compare_df = pd.DataFrame.from_dict(annualized_returns_compare, orient='index', columns=['Annualized Return'])
print(annualized_returns_compare_df)

my_pool_performance = MetricsCalculator('My Pool', pool_vs_allmarket['RETURN_MYPOOL'], pool_vs_allmarket['TRADE_DT'])
all_market_performance = MetricsCalculator('All Market', pool_vs_allmarket['RETURN_NXT'], pool_vs_allmarket['TRADE_DT'])
my_pool_performance.print_metrics()
all_market_performance.print_metrics()

'''流动性因子检验'''
my_pool_liquidity_test = my_pool[((my_pool['S_DQ_OPEN'] != my_pool['S_DQ_LIMIT']) & (my_pool['S_DQ_OPEN'] != my_pool['S_DQ_STOPPING']))]
my_pool_liquidity_test.reset_index(inplace=True, drop=True)
my_pool_liquidity_test_ind = data.assign_industry(my_pool_liquidity_test)
my_pool_liquidity_test_ind['S_DQ_TURN'] = my_pool_liquidity_test_ind['S_DQ_TURN'].astype(float)
my_pool_liquidity_test_ind['S_DQ_TURN_winsorized'] =  my_pool_liquidity_test_ind.groupby('TRADE_DT')['S_DQ_TURN'].transform(lambda x: winsorize(x, limits=[0.05, 0.05]))
my_pool_liquidity_test_ind['S_DQ_TURN_norm'] = my_pool_liquidity_test_ind.groupby('TRADE_DT')['S_DQ_TURN_winsorized'].transform(lambda x: zscore(x))

my_pool_liquidity_test_mvn = my_pool_liquidity_test_ind.groupby('TRADE_DT').apply(lambda x: data.mv_neutralize(x))
test_mp = FactorDecileAnalysis(my_pool_liquidity_test_mvn,5)
cleaned_df_decile_my_pool = test_mp.industry_neutralize_and_group()
ew_date_decile_my_pool = test_mp.calculate_average_daily_returns()
long_short_df_my_pool = test_mp.long_short_NAV()
icir_metrics_my_pool = test_mp.calculate_ic_metrics()
print('Liquidity factor metrics in my pool')
print(icir_metrics_my_pool)

all_market_liquidity_test = data_all[((data_all['S_DQ_OPEN'] != data_all['S_DQ_LIMIT']) & (data_all['S_DQ_OPEN'] != data_all['S_DQ_STOPPING'])) | \
                                     (data_all['S_DQ_TRADESTATUS'] != '停牌')]
all_market_liquidity_test.reset_index(inplace=True, drop=True)
all_market_liquidity_test_ind = data.assign_industry(all_market_liquidity_test)
all_market_liquidity_test_ind['S_DQ_TURN'] = all_market_liquidity_test_ind['S_DQ_TURN'].astype(float)
all_market_liquidity_test_ind['S_DQ_TURN_winsorized'] =  all_market_liquidity_test_ind.groupby('TRADE_DT')['S_DQ_TURN'].transform(lambda x: winsorize(x, limits=[0.05, 0.05]))
all_market_liquidity_test_ind['S_DQ_TURN_norm'] = all_market_liquidity_test_ind.groupby('TRADE_DT')['S_DQ_TURN_winsorized'].transform(lambda x: zscore(x))

all_market_liquidity_test_mvn = all_market_liquidity_test_ind.groupby('TRADE_DT').apply(lambda x: data.mv_neutralize(x))
test_am = FactorDecileAnalysis(all_market_liquidity_test_mvn, 5)
cleaned_df_decile_all_market = test_am.industry_neutralize_and_group()
ew_date_decile_all_market = test_am.calculate_average_daily_returns()
long_short_df_all_market = test_am.long_short_NAV()
icir_metrics_all_market = test_am.calculate_ic_metrics()

plt.figure(figsize=(12, 6))
plt.plot(long_short_df_all_market['TRADE_DT'], long_short_df_all_market['NAV_adj'], label = 'All Market')
plt.plot(long_short_df_my_pool['TRADE_DT'], long_short_df_my_pool['NAV_adj'], label = 'Long-Short Portfolio Adjusted (Exposure 1)')
plt.legend()
plt.show()

print('Liquidity factor metrics in my pool')
print(icir_metrics_my_pool)
print('Liquidity factor metrics in all market')
print(icir_metrics_all_market)

my_pool['Year'] = my_pool['TRADE_DT'].dt.year
data_all['Year'] = data_all['TRADE_DT'].dt.year
my_pool_CS010_avg = my_pool.groupby(['TRADE_DT', 'Year'])['S_DQ_AMOUNT'].mean().reset_index()
data_all_avg = data_all.groupby(['TRADE_DT', 'Year'])['S_DQ_AMOUNT'].mean().reset_index()
my_pool_CS010_avg['Pool'] = 'CS010'
data_all_avg['Pool'] = 'All'
combined_df = pd.concat([my_pool_CS010_avg, data_all_avg])

plt.figure(figsize=(12, 6))
sns.boxplot(x='Year', y='S_DQ_AMOUNT', hue='Pool', data=combined_df)
plt.title('Comparison of Daily Average Transaction Amounts by Year')
plt.xlabel('Year')
plt.ylabel('Daily Average Transaction Amount')
plt.legend(title='Stock Pool')
plt.show()

# 计算每年的平均成交额
yearly_avg_df = combined_df.groupby(['Year', 'Pool'])['S_DQ_AMOUNT'].mean().unstack()
print(yearly_avg_df)
