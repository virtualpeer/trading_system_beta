import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection._split import _BaseKFold

class PurgedKFold(_BaseKFold):
    def __init__(self, n_splits=3, tl=None, pctEmbargo=0.):
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

def cvScore(clf, X, y, sample_weight, scoring='accuracy', tl=None, cv=None, cvGen=None, pctEmbargo=None):
    if not (scoring in ['neg_log_loss', 'accuracy']):
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss, accuracy_score
    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, tl=tl, pctEmbargo=pctEmbargo)
    score = []
    for train, test in tqdm(list(cvGen.split(X))):
        fit = clf.fit(X.iloc[train], y.iloc[train], sample_weight=sample_weight.iloc[train].values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test])
            score_ = -log_loss(y.iloc[test], prob, sample_weight=sample_weight.iloc[test].values, labels=clf.classes_)
        else:
            pred = fit.predict(X.iloc[test])
            score_ = accuracy_score(y.iloc[test], pred, sample_weight=sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)


