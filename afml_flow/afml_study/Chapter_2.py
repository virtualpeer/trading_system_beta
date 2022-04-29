import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from tqdm import tqdm

def mad_outlier(y, thresh=3.):
    """"
    :param y: shape (N,1)
    :param thresh:
    :return: array
    """

    median = np.median(y)
    diff = np.sum((y-median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh

def get_bar_index(df, mode, unit=None):
    assert mode in ['time', 'tick', 'volume', 'dollar']
    df0 = df.reset_index().rename(columns={'dates': 'time'})
    num_days = (df0.time.values[-1] - df0.time.values[0]).astype('timedelta64[D]').astype(int)
    t, ts = df0['price' if mode == 'tick' else mode], 0
    idx, diff = [], []
    if mode == 'time':
        assert unit in [None, '1d']
        t, m = t.dt.date, '1d'
        idx.append(0)
        for i, (before, after) in enumerate(zip(t.values[:-1], t.values[1:]), 1):
            if after-before >= timedelta(days=1):
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
    return idx, (m, np.std(diff))

def get_target_df(df, start_date, num_days):
    return df.loc[np.datetime64(start_date):np.datetime64(start_date)+np.timedelta64(num_days,'D')]

def get_bar_df(df, mode, unit=None):
    assert mode in ['time', 'tick', 'volume', 'dollar']
    idx, (m, std) = get_bar_index(df, mode, unit)
    return df.iloc[idx].drop_duplicates(), (m, std)


def plot_sample_data(ref, sub, bar_type, *args, **kwargs):
    f, axes = plt.subplots(3, sharex=True, sharey=True, figsize=(10,7))
    ref.plot(*args, **kwargs, ax=axes[0], label='price')
    sub.plot(*args, **kwargs, ax=axes[0], ls='', marker='x', label=bar_type)
    axes[0].legend()

    ref.plot(*args, **kwargs, ax=axes[1], marker='o', label='price')
    sub.plot(*args, **kwargs, ax=axes[2], marker='x', label=bar_type, color='r', ls='')

    for ax in axes[1:]:
        ax.legend()

    plt.tight_layout()

def getTEvents(gRaw, upper, lower=None):
    if lower is None:
        lower = - upper
    assert (upper >=0 and lower <= 0)
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i, change in enumerate(diff.values[1:]):
        sPos, sNeg = max(0, sPos + change), min(0, sNeg + change)

        if sNeg < lower:
            sNeg = 0
            tEvents.append(i)

        if sPos > upper:
            sPos = 0
            tEvents.append(i)

    return gRaw.index[tEvents]


def get_ohlc(ref, sub):
    data = []
    for i in tqdm(range(len(sub))):
        if i < len(sub)-1:
            temp = ref.loc[sub.index[i]: sub.index[i+1]]
        else:
            temp = ref.loc[sub.index[i]:]
        o, h, l = sub[i], temp.dropna().max(), temp.dropna().min()
        data.append([sub.index[i], o,h,l])
    return pd.DataFrame(data, columns=['date', 'price','high','low']).set_index('date')



