import pandas as pd
import numpy as np
from afml.Chapter_20 import *

def returns(price):
    """
    :param price: pd.Series(float, index=np.datetime64)
    :return: pd.Series(ror, index=price.index[1:])
    """
    return pd.Series(np.array(price.values[1:]) / np.array(price.values[:-1]) - 1, index=price.index[1:])

def getDailyVol(price, span0=100):
    df0 = price.index.searchsorted(price.index - pd.Timedelta(days=1))
    df0 = df0[df0>0]
    df0 = pd.Series(price.index[df0], index=price.index[price.shape[0]-df0.shape[0]:])
    df0 = price.loc[df0.index] / price.loc[df0.values].values - 1
    df0 = df0.ewm(span=span0).std()
    return df0

def getVol(price, span0=100):
    """
    :param price: pd.Series(float(open price of each day), index=np.datetime64 but without repetition of dates)
    :param span0: ewm parameter
    :return: pd.Series(exponentially weighted moving standard deviation of daily returns, index='date', first element=NaN)
    """
    return pd.Series(np.array(price.values[1:]) / np.array(price.values[:-1]) - 1, index=price.index.date[1:]).ewm(span=span0).std()

def addVerticalBarrier(tEvents, price, numDays=None):
    """
    refer to snippet 3.4, in book, it resorts to pd.Timedelta(days=Numdays), but here we assume general-bars, which are not necessarily time-bars, so I set h=10
    :param tEvents: getTEvents(returns(volume_df.price), threshold) = pd.DateTimeIndex(sampled with cumsum)
    :param price: whole data before sampling, volume_df.price with index=np.datetime64
    :param horizon: vertical barrier size, initialization=10
    :return: Series with values of tl timestamp and index of tEvents timestamp
    """
    tl = price.index.searchsorted(tEvents) + 10 if numDays is None else price.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
    tl = tl[tl < len(price)]
    return pd.Series(price.index[tl], index=tEvents[:len(tl)])

def applyPtSlonTl(price, events, ptSl, molecule):
    """
    refer to the explanation before snippet 3.2
    :param price: volume_df.price
    :param events: index=tEvents, columns=['tl','target'(dailyvol), 'side']
    :param ptSl: horizontal bar width ratio
    :param molecule: a list with the subset of event indices that will be processed by a single thread
    :return: index=tEvents, columns=['pt', 'sl'], first time touching each barrier
    """
    events_ = events.loc[molecule]
    out = events_[['tl']].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0]*events_['trgt']
    else:
        pt = pd.Series(index=events.index) # NaNs
    if ptSl[1] > 0:
        sl = -ptSl[1]*events_['trgt']
    else:
        sl = pd.Series(index=events.index)

    for loc, tl in events_['tl'].fillna(price.index[-1]).iteritems():
        df0 = price[loc:tl]
        df0 = (df0 / price[loc] - 1) * events_.at[loc,'side']
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()

    return out

def getEvents(price, tEvents, ptSl, trgt, minRet, numThreads, tl=False, side=None):
    """
    refer to the explanation before and after snippet 3.3, with enhancements explained right before snippet 3.6
    :param price:
    :param tEvents:
    :param ptSl:
    :param trgt: dailyVol(horizontal bar unit width), index = price.index which contains tEvents
    :param minRet: at last dailyVol must be larger than this scalar
    :param numThreads:
    :param tl: index = tEvents
    :param side: side
    :return: index=subset of trgt.index(tEvents), columns=['tl','trgt', 'side'(if side is not None)] where 'tl' is not vertical bar timestamp but first-time for barrier touch
    """

    trgt = trgt.loc[trgt.index.isin(tEvents)]
    trgt = trgt[trgt > minRet]

    if tl is False:
        tl = pd.Series(pd.NaT, index=tEvents)

    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0],ptSl[0]]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:]

    events = pd.concat({'tl': tl, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    df0 = mpPandasObj(func=applyPtSlonTl, pdObj=('molecule', events.index), numThreads=numThreads, price=price, events=events, ptSl=ptSl_)
    events['tl'] = df0.dropna(how='all').min(axis=1)

    if side is None:
        events = events.drop('side', axis=1)

    return events

def getBins(events, price):
    """
    refer to the explanation before snippet 3.5
    :param events: returned value of getEvents, columns=['trgt','tl']
    :param price: volume_df.price
    :return: columns=['ret', 'bin']
    """
    events_ = events.dropna(subset=['tl'])
    px = events_.index.union(events_['tl'].values).drop_duplicates()
    px = price.reindex(px, method='bfill')
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['tl'].values].values / px.loc[events_.index] - 1
    if 'side' in events_:
        out['ret'] *= events_['side']
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_:
        out.loc[out['ret']<=0, 'bin'] = 0
    return out

def dropLabels(events, minPtc=.05):
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > minPtc or df0.shape[0] < 3:
            break
        print ('dropped label', df0.argmin(), df0.min())
        events = events[events['bin']!=df0.argmin()]
    return events






