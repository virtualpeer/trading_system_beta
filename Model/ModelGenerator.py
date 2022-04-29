from Engine.Core.Multiprocessing import *

class ModelGenerator:
    def __init__(self, events, price, numThreads, clfLastW):
        self.numCoEvents = mpPandasObj(func=self.__mpNumCoEvents, pdObj=('molecule', events.index), numThreads=numThreads, priceIdx=price.index, tl=events['tl'])
        self.sampleTw = mpPandasObj(func=self.__mpSampleTW, pdObj=('molecule', events.index), numThreads=numThreads, tl=events['tl'], numCoEvents=self.numCoEvents)
        sample_weight = pd.Series(name='weight', dtype=float)
        sample_weight = mpPandasObj(func=self.__mpSampleW, pdObj=('molecule', events.index), numThreads=numThreads, tl=events['tl'], numCoEvents=self.numCoEvents, price=price)
        sample_weight = sample_weight * self.__getTimeDecay(sample_weight, clfLastW)
        self.sample_weight = sample_weight * sample_weight.shape[0] / sample_weight.sum()

    def model(self):
        pass

    def __getTimeDecay(self, tW, clfLastW=1.):
        clfW = tW.sort_index().cumsum()
        slope = (1-clfLastW) / clfW.iloc[-1] if clfLastW >= 0 else 1./((clfLastW+1)*clfW.iloc[-1])
        const = 1 - slope*clfW.iloc[-1]
        clfW = const + slope*clfW
        clfW[clfW < 0] = 0
        print(f'constant: {const}, slope: {slope}')
        return clfW

    def __mpNumCoEvents(self, priceIdx, tl, molecule):
        """
        refer to the explanation on Snippet 4.1
        :param priceIdx:
        :param tl: time limit
        :param molecule: events.index if numThreads=1 or subset otherwise
        :return: compute c_t for t in price.index
        """

        tl = tl.fillna(priceIdx[-1])
        tl = tl[tl >= molecule[0]]
        tl = tl.loc[:tl[molecule].max()]

        iloc = priceIdx.searchsorted(np.array([tl.index[0], tl.max()]))
        count = pd.Series(0, index=priceIdx[iloc[0]:iloc[1]+1])
        for tIn, tOut in tl.iteritems():
            count.loc[tIn:tOut] += 1

        return count.loc[molecule[0]:tl[molecule].max()]

    def __mpSampleTW(self, tl, numCoEvents, molecule):
        """
        :param tl: events['tl']
        :param numCoEvents: pd.Series with index
        :param molecule: events.index if numThreads=1 or subset otherwise
        :return: compute u_i_bar for i in events.index
        """

        wght = pd.Series(index=molecule)

        for tIn, tOut in tl.loc[wght.index].iteritems():
            wght.loc[tIn] = (1./numCoEvents.loc[tIn:tOut]).mean()

        return wght

    def __mpSampleW(self, tl, numCoEvents, price, molecule):
        """

        :param tl: events['tl']
        :param numCoEvents: c_t for t in price.index
        :param price: volume / dollar bar price
        :param molecule: events.index if numThreads=1 or subset otherwise
        :return: compute \tilde{w_i} for i in events.index (refer to formula right before snippet 4.10)
        """
        ret = np.log(price).diff()
        wght = pd.Series(index=molecule)
        for tIn, tOut in tl.loc[wght.index].iteritems():
            wght.loc[tIn] = (ret.loc[tIn:tOut] / numCoEvents[tIn: tOut]).sum()
        return wght.abs()