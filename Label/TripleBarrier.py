from Label.Label import Label
import pandas as pd
from Engine.Core.Multiprocessing import *

class TripleBarrier(Label):
    def __init__(self, price, tEvents, label_config):
        self.price = price
        self._tEvents = tEvents
        self._vertical_barrier_dic = label_config['vertical barrier']
        self._ptSl = label_config['ptSl']
        self._minRet = label_config['minRet']
        self._numThreads = label_config['numThreads']
        self._tl = label_config['tl']
        self._side = label_config.get('side', None)

    @staticmethod
    def __addVerticalBarrier(tEvents, price, **vertical_barrier_dic):
        """
        refer to snippet 3.4, in book, it resorts to pd.Timedelta(days=Numdays), but here we assume general-bars, which are not necessarily time-bars, so I set h=10
        :param tEvents: getTEvents(returns(volume_df.price), threshold) = pd.DateTimeIndex(sampled with cumsum)
        :param price: whole data before sampling, volume_df.price with index=np.datetime64
        :param horizon: vertical barrier size, initialization=10
        :return: Series with values of tl timestamp and index of tEvents timestamp
        """
        tl = price.index.searchsorted(tEvents) + 10 if vertical_barrier_dic is None else price.index.searchsorted(tEvents + pd.Timedelta(**vertical_barrier_dic))
        tl = tl[tl < len(price)]
        return pd.Series(price.index[tl], index=tEvents[:len(tl)])

    @staticmethod
    def __applyPtSlonTl(price, events, ptSl, molecule):
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

    def __getEvents(self, trgt, side=None):
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

        events = self._tEvents
        tl = self.__addVerticalBarrier(tEvents=events, price=self.price,
                                       **self._vertical_barrier_dic)

        if self._tl is False:
            tl = pd.Series(pd.NaT, index=events)

        trgt = trgt.loc[trgt.index.isin(events) & trgt.index.isin(tl.index)]
        trgt = trgt[trgt > self._minRet]

        if side is None:
            side_, ptSl_ = pd.Series(1., index=trgt.index), [self._ptSl[0],self._ptSl[0]]
        else:
            side_, ptSl_ = side.loc[trgt.index], self._ptSl[:]

        events = pd.concat({'tl': tl, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
        df0 = mpPandasObj(func=self.__applyPtSlonTl, pdObj=('molecule', events.index),
                          numThreads=self._numThreads, price=self.price, events=events, ptSl=ptSl_)
        events['tl'] = df0.dropna(how='all').min(axis=1)

        if side is None:
            events = events.drop('side', axis=1)

        return events

    def getBins(self, trgt):
        """
        refer to the explanation before snippet 3.5
        :param events: returned value of getEvents, columns=['trgt','tl']
        :param price: volume_df.price
        :return: columns=['ret', 'bin']
        """
        events = self.__getEvents(trgt, self._side)
        events_ = events.dropna(subset=['tl'])
        px = events_.index.union(events_['tl'].values).drop_duplicates()
        px = self.price.reindex(px, method='bfill')
        out = pd.DataFrame(index=events_.index)
        out['ret'] = px.loc[events_['tl'].values].values / px.loc[events_.index] - 1
        if 'side' in events_:
            out['ret'] *= events_['side']
        out['bin'] = np.sign(out['ret'])
        if 'side' in events_:
            out.loc[out['ret']<=0, 'bin'] = 0
            out['side'] = events_['side']
        out['tl'] = events_.tl.loc[events_.index.isin(out.index)]
        out = out.reindex(columns=(['tl', 'ret', 'side', 'bin'] if 'side' in events_ else ['tl', 'ret', 'bin']))
        return out

    def dropLabels(self, events, minPtc=.05):
        while True:
            df0 = events['bin'].value_counts(normalize=True)
            if df0.min() > minPtc or df0.shape[0] < 3:
                break
            print ('dropped label', df0.argmin(), df0.min())
            events = events[events['bin']!=df0.argmin()]
        return events