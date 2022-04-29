from Strategy.Strategy import Strategy
import pandas as pd
import numpy as np
from Engine.Core.Multiprocessing import *

class AverageStrategy(Strategy):
    def __init__(self, pred, prob, out, price, numClasses, config):
        super().__init__(pred, prob, out, price, config)
        self.numClasses = numClasses
        self.price = price
        self.step_size = self._config['stepSize']
        self.numThreads = self._config['numThreads']

    def evaluate(self):
        if not (self.scoring in ['ror', 'accuracy']):
            raise Exception('wrong strategy scoring method.')

        action = self.action()
        action.sort_index(inplace=True)
        if not (True in pd.isnull(self.out['tl'])):
            assert action.values[-1] == 0

        if self.scoring == 'ror':
            relevant = self.price.reindex(action.index[:-1])
            score = ((relevant.iloc[1:].values / relevant.iloc[:-1] - 1) * action + 1).product()

        else:
            from sklearn.metrics import accuracy_score
            relevant = action.index.isin(self.out.index)
            score = accuracy_score(self.out.bin, np.sign(action.loc[relevant]))

        return score

    def action(self, **kwargs):
        from scipy.stats import norm
        events = self.out
        if not self.prob.shape[0]:
            return pd.Series()
        signal0 = (self.prob - 1. / self.numClasses) / (self.prob * (1. - self.prob)) ** 0.5
        signal0 = self.pred * (2 * norm.cdf(signal0) - 1)
        if 'side' in events:
            signal0 *= events.loc[signal0.index, 'side']
        df0 = signal0.to_frame('signal').join(events[['tl']], how='left')
        df0 = self.__avgActiveSignals(df0, self.numThreads)
        signal1 = self.__discreteSignal(signal0=df0, stepSize=self.step_size)
        return signal1

    def __avgActiveSignals(self, signals, numThreads):
        tPnts = set(signals['tl'].dropna().values).union(signals.index.values)
        tPnts = list(tPnts)
        tPnts.sort()
        out = mpPandasObj(func=self.__mpAvgActiveSignals, pdObj=('molecule', tPnts), numThreads=numThreads, signals=signals)
        return out

    @staticmethod
    def __mpAvgActiveSignals(signals, molecule):
        out = pd.Series()
        for loc in molecule:
            df0 = (signals.index.values <= loc) & ((loc < signals['tl'].values) | pd.isnull(signals['tl']))
            act = signals.loc[df0].index
            if len(act) > 0:
                out[loc] = signals.loc[act, 'signal'].mean()
            else:
                out[loc] = 0
        return out

    @staticmethod
    def __discreteSignal(signal0, stepSize):
        signal1 = (signal0 / stepSize).round() * stepSize
        signal1[signal1 > 1] = 1
        signal1[signal1 < -1] = -1
        return signal1




