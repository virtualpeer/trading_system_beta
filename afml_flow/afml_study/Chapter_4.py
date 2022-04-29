import numpy as np
import pandas as pd
def mpNumCoEvents(priceIdx, tl, molecule):
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

def mpSampleTW(tl, numCoEvents, molecule):
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

def getIndMatrix(barIx, tl):
    """
    refer to chapter 4.5.3 a numerical example
    :param barIx: price.index
    :param tl: events['tl']
    :return:
    """
    indM = pd.DataFrame(0, index=barIx, columns=range(tl.shape[0]))
    for i, (t0, t1) in enumerate(tl.iteritems()):
        indM.loc[t0:t1, i] = 1

    return indM

def getAvgUniqueness(indM):
    c = indM.sum(axis=1)
    u = indM.div(c, axis=0)
    avgU = u[u>0].mean()
    return avgU

def seqBootstrap(indM, sLength=None):
    """
    refer to Chapter 4.5.3 numerical example!
    """
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series()
        for i in indM:
            indM_ = indM[phi+[i]]
            # 새로 넣은 i column에 대한 AvgUniqueness는 getAvgUniquenses에 의해 return되는 series의 맨 끝 값에 있음!
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU / avgU.sum()
        phi += [np.random.choice(indM.columns, p=prob)]
        print(phi)
    return phi

def mpSampleW(tl, numCoEvents, price, molecule):
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