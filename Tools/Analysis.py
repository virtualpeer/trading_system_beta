import pandas as pd
import numpy as np
from tqdm import tqdm
import os


def featImpMDI(fit, featNames):
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan)
    imp = pd.concat({'mean': df0.mean(), 'std': df0.std() * df0.shape[0]**(-0.5)}, axis=1)
    imp /= imp['mean'].sum()
    return imp


def featImp(clf, X, y, sample_weight, scoring, cvGen, sfi=False):

    if scoring not in ['accuracy', 'neg_log_loss']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss, accuracy_score

    imp_MDA_original, imp_MDA_permute = pd.Series(), pd.DataFrame(columns=X.columns)

    imp_MDI = pd.DataFrame(index=X.columns)
    imp_MDI_data = []

    imp_SFI = pd.DataFrame(columns=X.columns)

    for i, (train, test) in tqdm(list(enumerate(cvGen.split(X=X)))):
        X0, y0, w0 = X.iloc[train], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test], y.iloc[test], sample_weight.iloc[test]

        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)
        imp_MDI_data.append(featImpMDI(fit=fit, featNames=X.columns))

        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            imp_MDA_original.loc[i] = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
        else:
            pred = fit.predict(X1)
            imp_MDA_original.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_.loc[:,j].values)
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                imp_MDA_permute.loc[i,j] = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
            else:
                pred = fit.predict(X1_)
                imp_MDA_permute.loc[i,j] = accuracy_score(y1, pred, sample_weight=w1.values)

        if sfi:
            for j in X.columns:
                fit = clf.fit(X=X0[[j]], y=y0, sample_weight=w0.values)
                if scoring == 'neg_log_loss':
                    prob = fit.predict_proba(X1[[j]])
                    imp_SFI.loc[i,j] = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
                else:
                    pred = fit.predict(X1[[j]])
                    imp_SFI.loc[i,j] = accuracy_score(y1, pred, sample_weight=w1.values)

    concat = pd.concat(imp_MDI_data, axis=1)
    imp_MDI.loc[:, 'mean'] = concat.loc[:, 'mean'].mean(axis=1)
    imp_MDI.loc[:, 'std'] = (((concat.loc[:, 'std'] ** 2).sum(axis=1)) ** 0.5) / (concat.shape[1]/2)

    imp_MDA = (-imp_MDA_permute).add(imp_MDA_original, axis=0)

    if scoring == 'neg_log_loss':
        imp_MDA /= -imp_MDA_permute
    else:
        imp_MDA /= 1-imp_MDA_permute

    imp_MDA = pd.concat({'mean': imp_MDA.mean(), 'std': imp_MDA.std()*imp_MDA.shape[0]**(-0.5)}, axis=1)
    imp_SFI = pd.concat({'mean': imp_SFI.mean(), 'std': imp_SFI.std()*imp_SFI.shape[0]**(-0.5)}, axis=1) if sfi \
        else pd.DataFrame(np.nan, index=X.columns, columns=['mean', 'std'])

    pca = PCA_rank(X)

    return imp_MDI, imp_MDA, imp_SFI, pca, imp_MDA_original


def get_eVec(dot, varThres):
    eVal, eVec = np.linalg.eigh(dot)
    idx = eVal.argsort()[::-1]
    eVal, eVec = eVal[idx], eVec[:, idx]
    eVal = pd.Series(eVal, index=['PC_'+str(i+1) for i in range(eVal.shape[0])])
    eVec = pd.DataFrame(eVec, index=dot.index, columns=eVal.index)
    eVec = eVec.loc[:,eVal.index]
    cumVar = eVal.cumsum() / eVal.sum()
    dim = cumVar.values.searchsorted(varThres)
    eVal, eVec = eVal.iloc[:dim+1], eVec.iloc[:,:dim+1]
    return eVal, eVec


def orthoFeats(dfX, varThres=0.95):
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)
    eVal, eVec = get_eVec(dot, varThres)
    dfP = np.dot(dfZ, eVec)
    return  dfP


def PCA_rank(dfX):
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)
    dot1 = np.dot(dfZ.T, dfZ)
    eVal1, eVec1 = np.linalg.eig(dot1)

    perm = np.random.permutation(dfZ.columns)
    dfZ = dfZ.reindex(perm, axis=1)
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=perm, columns=perm)
    eVal, eVec = np.linalg.eig(dot)

    return pd.Series(eVal.shape[0] - eVal.argsort().argsort(), index=dfX.columns, name='PCA_rank')


def featImp_rank(imp, name):
    assert 'mean' in imp
    return pd.Series(imp.shape[0] - imp['mean'].argsort().argsort(), name=name)


