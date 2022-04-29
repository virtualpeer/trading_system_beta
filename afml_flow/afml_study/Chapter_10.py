import pandas as pd
from scipy.stats import norm
from afml.Chapter_20 import *

def getSignal(events, stepSize, prob, pred, numClasses, numThreads, **kwargs):
    """
    :param events:
    :param stepSize:
    :param prob: pd.Series with index=events.index
    :param pred: pd.Series with index=events.index
    :param numClasses: fit._classes
    :param numThreads:
    :param kwargs:
    :return:
    """
    if not prob.shape[0]:
        return pd.Series()
    signal0 = (prob - 1. / numClasses) / (prob * (1. - prob)) ** 0.5
    signal0 = pred * (2 * norm.cdf(signal0) - 1)
    if 'side' in events:
        signal0 *= events.loc[signal0.index, 'side']
    df0 = signal0.to_frame('signal').join(events[['tl']], how='left')
    df0 = avgActiveSignals(df0, numThreads)
    signal1 = discreteSignal(signal0=df0, stepSize=stepSize)
    return signal1


def avgActiveSignals(signals, numThreads):
    tPnts = set(signals['tl'].dropna().values).union(signals.index.values)
    tPnts = list(tPnts)
    tPnts.sort()
    out = mpPandasObj(func=mpAvgActiveSignals, pdObj=('molecule', tPnts), numThreads=numThreads, signals=signals)
    return out

def mpAvgActiveSignals(signals, molecule):
    out = pd.Series()
    for loc in molecule:
        df0 = (signals.index.values <= loc) & ((loc < signals['tl'].values) | pd.isnull(signals['tl']))
        act = signals.loc[df0].index
        if len(act) > 0:
            out[loc] = signals.loc[act,'signal'].mean()
        else:
            out[loc] = 0
    return out

def discreteSignal(signal0, stepSize):
    signal1 = (signal0 / stepSize).round()*stepSize
    signal1[signal1 > 1] = 1
    signal1[signal1 < -1] = -1
    return signal1

