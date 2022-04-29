from sklearn.model_selection._split import _BaseKFold
import pandas as pd
import numpy as np

class PurgedKFold(_BaseKFold):
    def __init__(self, cv_config, tl=None):
        self._cv_config = cv_config
        n_splits = self._cv_config.get('n_splits', 3)
        pctEmbargo = self._cv_config.get('pctEmbargo', 0.)
        if not isinstance(tl, pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.tl = tl
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if False in X.index == self.tl.index:
            raise ValueError('X and ThruDateValues must have the same index')
        indices = np.arange(X.shape[0])
        mbrg = int(self.pctEmbargo*X.shape[0])
        test_starts = [(i[0], i[-1]+1) for i in np.array_split(indices, self.n_splits)]
        for i, j in test_starts:
            t0 = self.tl.index[i]
            test_indices = indices[i:j]
            maxTlIdx = self.tl.index.searchsorted(self.tl[test_indices].max())
            train_indices = self.tl.index.searchsorted(self.tl[self.tl <= t0].index)
            if maxTlIdx < X.shape[0]:
                train_indices = np.concatenate([train_indices, indices[maxTlIdx+mbrg:]])
            yield train_indices, test_indices

class PurgedWF(_BaseKFold):
    def __init__(self, cv_config, tl=None):
        self._cv_config = cv_config
        n_splits = self._cv_config.get('n_splits', 3)
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        if not isinstance(tl, pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        self.warmup = self._cv_config.get('warmup', 0.5)
        self.tl = tl


    def split(self, X, y=None, groups=None):
        if False in X.index == self.tl.index:
            raise ValueError('X and ThruDateValues must have the same index')

        indices = np.arange(X.shape[0])
        relevant = round(self.warmup*len(indices))
        test_set = [(i[0], i[-1]+1) for i in np.array_split(indices[relevant+1:], self.n_splits)]
        for i, j in test_set:
            test_indices = indices[i:j]
            train_indices = self.tl.index.searchsorted(self.tl.loc[self.tl <= self.tl.iloc[test_indices].index.min()].index)
            yield train_indices, test_indices