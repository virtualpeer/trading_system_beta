from Chapter_2 import *
from Chapter_3 import *
from Chapter_4 import *
from Chapter_20 import *
import os, time, platform
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook

# Workflow:
# get_bar_df 이용해서 volume이나 dollar bar 샘플링, getTEvents 이용해서 CUMSUM 샘플링, addVerticalBarrier 이용해서 tl 구하기, mpPandasObj(func=getEvents,...) 이용하여 events 구하기, getBins 이용하여 labeling
# mpPandasObj(func=mpNumCoEvents,...) 이용하여 (c_t for t in price.index) 구하기, mpPandasObj(func=mpSampleTW, .., numCoEvents=numCoEvents) 이용하여 (u_i bar for i in events.index) 구하기
# getIndMatrix, seqBootstrap 이용하여 sequential bootstrap 진행, mpPandasObj(func=mpSampleW,...) 이용하여 각 event에 줄 weight을 구함.

txt_path = 'C:\\Users\\SJ_Son\\Desktop'
df = pd.read_csv(os.path.join(txt_path, 'S&P_data.csv'), parse_dates=['dates'])
df.set_index('dates', inplace=True)
assert not (True in df.index.duplicated())
df = df.loc[~mad_outlier(df.price.values.reshape(-1,1))]
dollar_df, (unit, std) = get_bar_df(df, 'dollar', 1_000_000)
print(f'original bar: \n {df}')
print(f'dollar bar: \n {dollar_df}')

price = dollar_df.price.copy(deep=True)
dailyVol = getDailyVol(price)

# f, ax = plt.subplots()
# dailyVol.plot(ax=ax)
# ax.axhline(dailyVol.mean(), ls='--', color='red')

tEvents = getTEvents(np.log(price), upper=np.log(1+dailyVol.mean()), lower=np.log(1-dailyVol.mean()))
print(f'number sampled using CUMSUM: {tEvents.shape[0]}, original number in dollar bar: {price.shape[0]}')
tl = addVerticalBarrier(tEvents, price, 10)
ptSl = [1,1]
target = dailyVol
minRet = 0.001

if platform.system() == 'Windows':
    cpus = 1
else:
    cpus = cpu_count() - 1
print(f'daily volatility: \n {dailyVol} \n time limit: \n {tl}')

numThreads = cpus
events = getEvents(price, tEvents, ptSl, target, minRet, numThreads=numThreads, tl=tl)
out = getBins(events, price)
print(f'label: \n {out} \n {out.bin.value_counts()}')
numCoEvents = mpPandasObj(func=mpNumCoEvents, pdObj=('molecule', events.index), numThreads=numThreads, priceIdx=price.index, tl=events['tl'])

# plt.figure(figsize=(20,10))
# plt.xlabel('Time')

# ax1, ax2 = numCoEvents.plot(color='blue', grid=True, label='CoEvents'), dailyVol.plot(color='red', grid=True, secondary_y=True, label='daily volatility')
# ax1.legend(loc=1)
# ax2.legend(loc=2)
# plt.show()

sampleTw = mpPandasObj(func=mpSampleTW, pdObj=('molecule', events.index), numThreads=numThreads, tl=events['tl'], numCoEvents=numCoEvents)
print(f'mean average uniqueness: {sampleTw.mean()}, which is too high!')
sampleTw.plot.hist()
plt.show()

# Questions: takes too much time to bootstrap and is it necessary to consider making i.i.d sequence (uniqueness score is already too high)
# print(events.shape[0])
# indM = getIndMatrix(price.index, events['tl'])
# phi = seqBootstrap(indM, 10)
# print(events.index[phi])


sample_weight = pd.Series(name='w', dtype=float)
sample_weight = mpPandasObj(func=mpSampleW, pdObj=('molecule', events.index), numThreads=numThreads, tl=events['tl'], numCoEvents=numCoEvents, price=price)
sample_weight = sample_weight*sample_weight.shape[0]/sample_weight.sum()
print(f'sample_weight: {sample_weight}')
