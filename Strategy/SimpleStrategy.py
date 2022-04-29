from Strategy.Strategy import Strategy
import pandas as pd
import numpy as np
from tqdm import tqdm


class SimpleStrategy(Strategy):

    def action(self):
        relevant = self.__action(self.out, self.pred.index)
        return super().action().iloc[relevant]

    @staticmethod
    def __action(out, test_index):
        current, tl_ = test_index[0], out['tl'].loc[test_index]
        temp = [current]
        while True:
            tl_ = tl_.loc[tl_[current]:]
            if tl_.empty:
                break
            current = tl_.index[0]
            temp.append(current)
        return test_index.searchsorted(temp)
