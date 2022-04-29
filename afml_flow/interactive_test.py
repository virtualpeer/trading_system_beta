from afml_study.Chapter_2 import *
from afml_study.Chapter_3 import *
from afml_study.Chapter_4 import *
from afml_study.Chapter_6 import *
from afml_study.Chapter_7 import *
from afml_study.Chapter_20 import *
import os
import matplotlib.pyplot as plt


txt_path = r'C:\Users\jaehkim\Desktop\tick data'
df = pd.read_csv(os.path.join(txt_path, 'S&P_data.csv'), parse_dates=['dates'])
df.set_index('dates', inplace=True)
assert not (True in df.index.duplicated())
df = df.loc[~mad_outlier(df.price.values.reshape(-1,1))]
dollar_df, (unit, std) = get_bar_df(df, 'dollar', 1_000_000)
print (dollar_df, unit, std)


dollar_df_price = dollar_df.price.copy(deep=True)
dailyVol = getDailyVol(dollar_df_price)
tEvents = getTEvents(np.log(dollar_df_price), upper=np.log(1+dailyVol.mean()), lower=np.log(1-dailyVol.mean()))
tl = addVerticalBarrier(tEvents, dollar_df_price,10)
target = dailyVol
print(f'daily volatility: \n {dailyVol} \n \n sampled datetime index: \n {tEvents} \n time limit: \n {tl}')


numThreads = 1
events = getEvents(dollar_df_price, tEvents, ptSl=[1,1], trgt=target, minRet=0.001, numThreads=numThreads, tl=tl, side=None)
out = getBins(events, dollar_df_price)
print(f'events: \n {events} \n label: \n {out}')
print(getBins(getEvents(dollar_df_price, tEvents, ptSl=[1,1], trgt=target, minRet=0.001, numThreads=numThreads, tl=tl, side=out.bin), dollar_df_price).bin.value_counts())


dataset = pd.concat([get_ohlc(df.price, dollar_df_price.loc[tEvents]), out.bin], axis=1).dropna(subset=['bin'])
print(f'dataset: \n {dataset}')

numCoEvents = mpPandasObj(func=mpNumCoEvents, pdObj=('molecule', events.index), numThreads=numThreads, priceIdx=dollar_df_price.index, tl=events['tl'])
sampleTw = mpPandasObj(func=mpSampleTW, pdObj=('molecule', events.index), numThreads=numThreads, tl=events['tl'], numCoEvents=numCoEvents)
print(f'number of concurrent events at time t for each t in price.index: \n {numCoEvents} \n uniqueness score for sample i for each i in tEvents: \n {sampleTw}')

sampleTw.plot.hist()
sampleTw.describe()

sample_weight = pd.Series(name='weight', dtype=float)
sample_weight = mpPandasObj(func=mpSampleW, pdObj=('molecule', events.index), numThreads=numThreads, tl=events['tl'], numCoEvents=numCoEvents, price=dollar_df_price)
sample_weight = sample_weight*sample_weight.shape[0]/sample_weight.sum()
print(sample_weight)
print(False in sample_weight.index == dataset.index == events['tl'].index, len(sample_weight))

split_year = '2019'
score = sampleTw
training_sample_weight, test_sample_weight = sample_weight.loc[:str(int(split_year)-1)], sample_weight.loc[split_year:]
training_set, test_set = dataset.loc[:str(int(split_year)-1)], dataset.loc[split_year:]

clf = bagging(2, n_estimators=1000, max_samples=score.mean(), oob_score=True)
clf.fit(X=training_set.iloc[:,:-1], y=training_set.iloc[:,-1], sample_weight=training_sample_weight)

print(f'out-of-bag sample score: {clf.oob_score_}')
print(f'train data accuracy: {clf.score(X=training_set.iloc[:,:-1], y=training_set.iloc[:,-1], sample_weight=training_sample_weight)}')
print(f'test data accuracy: {clf.score(X=test_set.iloc[:,:-1], y=test_set.iloc[:,-1], sample_weight=test_sample_weight)}')


from sklearn.metrics import precision_recall_fscore_support
labels=[-1,1]
y_pred = clf.predict(X=test_set.iloc[:,:-1])
print(test_set.iloc[:,-1].value_counts())
p, r, f, s = precision_recall_fscore_support(y_true=test_set.iloc[:,-1], y_pred=y_pred, labels=labels)
for result in zip(labels,p,r,f,s):
    print(f'label {result[0]}: \n precision, recall, f_score, support: {result[1:]}')
    
print(len(training_set)+len(test_set)==dataset.shape[0], training_set, test_set)

for n_split in range(2,11):
    print(cvScore(clf, X=dataset.iloc[:,:-1], y=dataset.iloc[:,-1], sample_weight=sample_weight, tl=events['tl'], cv=n_split, pctEmbargo=0.01))