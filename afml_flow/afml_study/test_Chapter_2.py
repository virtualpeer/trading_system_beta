from ToCSV import db_to_csv, txt_to_csv
from Chapter_2 import *
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


db_path, txt_path, file_path = 'C:\\Users\\SJ_Son\\Documents\\GitHub\\Kiwoom_datareader\\', 'C:\\Users\\SJ_Son\\Desktop', 'C:\\Users\\SJ_Son\\Documents\\GitHub\\AssetAllocation\\stat_assetallocation\\Lab\\traditional_assetallocation'
# db_to_csv(db_path, 304940)
# db_to_csv(db_path, '005930')
# txt_to_csv(txt_path)

# samsung = pd.read_csv(os.path.join(txt_path, '304940.csv'))
# nasdaq = pd.read_csv(os.path.join(txt_path, '005930.csv'))
df = pd.read_csv(os.path.join(txt_path, 'S&P_data.csv'), parse_dates=['dates'])
df.set_index('dates', inplace=True)

rets = pd.read_csv(os.path.join(file_path, 'rets.csv'), parse_dates=['Date'])
rets.set_index('Date', inplace=True)

mad = mad_outlier(df.price.values.reshape(-1,1))
df = df.loc[~mad]

start_date = '2010-10-29'
target_df = get_target_df(df, start_date, 200)
time_df, others_time = get_bar_df(target_df, 'time')
tick_df, others_tick = get_bar_df(target_df, 'tick')
volume_df, others_volume = get_bar_df(target_df, 'volume')

print(get_ohlc(df.price, volume_df.price.loc[getTEvents(volume_df.price, upper=0.005)]))

# print(others_volume, others_tick, len(time_df), len(tick_df), len(volume_df))
# f, axes = plt.subplots(3)
# ror_time = pd.Series(((np.array(time_df.price.values[1:]) / np.array(time_df.price.values[:-1]))-1)*100)
# ror_volume = pd.Series(((np.array(volume_df.price.values[1:]) / np.array(volume_df.price.values[:-1]))-1)*100)
# ror_tick = pd.Series(((np.array(tick_df.price.values[1:]) / np.array(tick_df.price.values[:-1]))-1)*100)
#
# print(rets.SPY.iloc[(ind:=rets.index.get_loc(np.datetime64(start_date))):ind+len(time_df)].mean()*100, ror_time.mean(), ror_tick.mean(), ror_volume.mean())
#
#
# ror_time.plot.hist(bins=50, ax=axes[0], title='time bar return')
# ror_volume.plot.hist(bins=50, ax=axes[1], title='volume bar return')
# ror_tick.plot.hist(bins=50, ax=axes[2], title='tick bar return')
#
# plt.show()
#
# plot_sample_data(target_df.price, volume_df.price, 'volume bar')
# plt.show()