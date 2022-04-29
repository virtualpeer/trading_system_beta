import numpy as np
from datetime import timedelta


class Preprocessor:
    def __init__(self, target_df, preprocessor_config):
        self._target_df = target_df
        self._unit = preprocessor_config.get('unit', None)
        self._remove_outlier = preprocessor_config['remove_outlier']
        self._mode = preprocessor_config['mode']
        if self._remove_outlier:
            self._target_df = self._target_df.loc[~self.__mad_outlier(self._target_df.price.values.reshape(-1, 1)
                                                                      if 'price' in self._target_df else
                                                                      self._target_df.close.values.reshape(-1, 1))]

    def __mad_outlier(self, y, thresh=3.):
        """"
        :param y: shape (N,1)
        :param thresh:
        :return: array
        """

        median = np.median(y)
        diff = np.sum((y - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        modified_z_score = 0.6745 * diff / med_abs_deviation
        return modified_z_score > thresh

    def __get_bar_index(self, df, mode, unit):
        df0 = df.reset_index().rename(columns={'dates': 'time'})
        dates = df0.time.dt.date.values
        num_days = (dates[-1] - dates[0]).days
        t, ts = df0['price' if mode == 'tick' else mode], 0
        idx, diff = [], []
        if mode == 'time':
            assert unit in [None, '1d']
            t, m = t.dt.date, '1d'
            idx.append(0)
            for i, (before, after) in enumerate(zip(t.values[:-1], t.values[1:]), 1):
                if after - before >= timedelta(days=1):
                    idx.append(i)
            diff.append(0)
        elif mode == 'tick':
            m = len(df0) // num_days if unit is None else unit
            for i, _ in enumerate(t):
                ts += 1
                if ts == m:
                    ts = 0
                    idx.append(i)
            diff.append(0)
        else:
            m = t.values.sum() // num_days if unit is None else unit
            for i, x in enumerate(t):
                ts += x
                if ts >= m:
                    idx.append(i)
                    diff.append(ts)
                    ts = 0
        return idx, ts, (m, np.std(diff))


    def make_bar_df(self):
        if not self._mode in ['time', 'tick', 'volume', 'dollar']:
            raise ValueError('mode must be time, tick, volume, or dollar')
        idx, ts, (m, std) = self.__get_bar_index(self._target_df, self._mode, self._unit)
        return self._target_df.iloc[idx].drop_duplicates(), ts, (m, std)


