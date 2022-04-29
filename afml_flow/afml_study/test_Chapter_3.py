from Chapter_2 import *
from Chapter_3 import *
import os
import pandas as pd
import numpy as np


txt_path = 'C:\\Users\\SJ_Son\\Desktop'
df = pd.read_csv(os.path.join(txt_path, 'S&P_data.csv'), parse_dates=['dates'])
df.set_index('dates', inplace=True)

# df = df.loc[~mad_outlier(df.price.values.reshape(-1,1))]
volume_df, (unit, std) = get_bar_df(df, 'volume', int(df.volume.sum() / len(df)))
time_df, others = get_bar_df(df, 'time')
dailyVol = getDailyVol(volume_df.price)

# volume bar의 return의 합이 너무 커지거나 작아지면 return의 평균값이 바뀌었다 볼 수 있음, 그 threshold는 평균 volatility에 대한 함수여야 할 것. 연습 예제에서는 아래와 같이 잡음.
# cumsum-filtering
price = volume_df.price.copy(deep=True)
tEvents = getTEvents(returns(price), dailyVol.mean() + 0.5*dailyVol.std())
tl = addVerticalBarrier(tEvents, price, 10)

ptSl = [1,1]
target = dailyVol
minRet = 0.001

numThreads = 1

events = getEvents(price, tEvents, ptSl, target, minRet, numThreads=numThreads, tl=tl)
out = getBins(events, price)


print(f'number sampled using CUMSUM: {tEvents.shape[0]}, original number in volume bar: {price.shape[0]} \n events:  {events} \n label: \n {out} \n {out.bin.value_counts()}')
print(f'what happens here: \n {(strange := getBins(getEvents(price, tEvents, ptSl=[1,1], trgt=target, minRet=0.001, numThreads=numThreads, tl=tl, side=out.bin), price))}'
      f'{strange.bin.value_counts()}')








