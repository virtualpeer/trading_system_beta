import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
import talib
from talib import MA_Type
from talib.abstract import *


def get_frequency(ref, sub_index):
    assert not (False in sub_index.isin(ref.index))
    dates = np.unique(ref.index.date)
    event_dates = sub_index.date
    temp = pd.Series(0, index=dates).sort_index()
    temp.loc[temp.index.isin(event_dates)] = pd.Series(event_dates).value_counts().sort_index()
    return temp.describe()


def get_return(ref, tEvents):
    price = ref.loc[tEvents.union([ref.index[0]])]
    return pd.Series(price.iloc[1:] / price.values[:-1] - 1, name='return')


def get_ask_bid_spread(ref, tEvents):
    return pd.Series((ref.ask - ref.bid).loc[tEvents], name='spread')


def get_vol(ref, span0=100, **kwargs):
    df0 = ref.index.searchsorted(ref.index - pd.Timedelta(**kwargs))
    df0 = df0[df0 > 0]
    df0 = pd.Series(ref.index[df0], index=ref.index[ref.shape[0] - df0.shape[0]:])
    try:
        df0 = ref.loc[df0.index] / ref.loc[df0.values].values - 1
        df0 = df0.ewm(span=span0).std()
    except ValueError:
        return f'Error! get_vol error occurred again, here is the information \n {df0}, \n {df0.index}, ' \
               f'\n {df0.values}, \n {ref.loc[df0.index]}, \n {ref.loc[df0.values]}, \n {ref.loc[df0.values].values}'
    return df0


class TALib:
    @staticmethod
    def convert(data, dates, *names):
        if isinstance(data, list):
            return [pd.Series(datum, index=dates, name=name) for datum, name in zip(data, names)]
        else:
            return [pd.Series(data, index=dates, name=names[0])]

    def get_features(self, features_list):
        if features_list is None:
            features_list = self.__dict__.keys()
        return pd.concat([getattr(self, '_'.join(feature.split())) for feature in features_list], axis=1)

    def set(self, dic, dates):
        for attributes, param in dic.items():
            attr_ = list(attributes) if isinstance(attributes, tuple) else [attributes]
            for attribute, value in zip(attr_, self.convert(param, dates, *attr_)):
                setattr(self, '_'.join(attribute.split()), value)

class TALibBase(TALib):
    def __init__(self, df, tEvents):
        temp = self.get_ohlcv(df, tEvents)

        self.open, self.high, self.low, self.close, self.volume = temp.open, temp.high, temp.low, temp.close, \
                                                                  temp.volume

        self.dic = {'open' : np.array(temp.open), 'high' : np.array(temp.high), 'low' : np.array(temp.low),
                    'close' : np.array(temp.close), 'volume' : np.array(temp.volume)}
    @staticmethod
    def get_ohlcv(ref, tEvents):
        assert not (False in tEvents.isin(ref.index) and isinstance(ref, pd.DataFrame))
        data = []
        ref = ref.sort_index()
        if 'price' in ref:
            for i in tqdm(range(len(tEvents))):
                relevant = ref.loc[tEvents[i - 1]: tEvents[i]] if i else ref.loc[:tEvents[i]]
                o, h, l, c, v = relevant.price.iloc[0], relevant.price.dropna().max(), relevant.price.dropna().min(), \
                                relevant.price.iloc[-1], relevant.volume.sum()
                data.append([tEvents[i], o, h, l, c, v])
        elif 'close' in ref:
            assert not (False in pd.Index(['open', 'high', 'low', 'close']).isin(ref.columns))
            for i in tqdm(range(len(tEvents))):
                relevant = ref.loc[tEvents[i - 1]:tEvents[i]] if i else ref.loc[:tEvents[i]]
                data.append([tEvents[i], relevant.close.iloc[0], relevant.iloc[1:, :].high.max(),
                             relevant.iloc[1:, :].low.min(), relevant.close.iloc[-1],
                             relevant.iloc[1:, :].volume.sum()])
        else:
            raise Exception('wrong data')
        return pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume']).set_index('date')

class TALibOverlap(TALib):
    def __init__(self, high, low, close, dates):
        #1
        dic = {'dema': DEMA(close), 'ema': EMA(close),
               'ht_trendline': HT_TRENDLINE(close), 'kama': KAMA(close), 'ma': MA(close),
               'midpoint': MIDPOINT(close), 'midprice': MIDPRICE(high, low), 'sar': SAR(high, low),
               'sarext': SAREXT(high, low), 'sma': SMA(close), 't3': T3(close), 'tema': TEMA(close),
               'trima': TRIMA(close), 'wma': WMA(close)}
        dic.update({k:v for k, v in zip(('upperband', 'middleband', 'lowerband'), BBANDS(close))})
        dic.update({k: v for k, v in zip(('mama', 'fama'), MAMA(close))})

        # #X2
        dic.update(
            {'dema2x': DEMA(close, timeperiod=60), 'ema2x': EMA(close, timeperiod=60),
             'ht_trendline2x': HT_TRENDLINE(close), 'kama2x': KAMA(close, timeperiod=60),
             'ma2x': MA(close, timeperiod=60),
             'midpoint2x': MIDPOINT(close, timeperiod=28), 'midprice2x': MIDPRICE(high, low, timeperiod=28),
             'sar2x': SAR(high, low, acceleration=0.04, maximum=0.4),
             'sarext2x': SAREXT(high, low, accelerationinitlong=0.04, accelerationlong=0.04,
                              accelerationmaxlong=0.4, accelerationinitshort=0.04, accelerationshort=0.04,
                              accelerationmaxshort=0.4),
             'sma2x': SMA(close, timeperiod=30), 't32x': T3(close, timeperiod=10),
             'tema2x': TEMA(close, timeperiod=60),
             'trima2x': TRIMA(close, timeperiod=60), 'wma2x': WMA(close, timeperiod=60)}
        )
        dic.update({k: v for k, v in zip(('upperband2x', 'middleband2x', 'lowerband2x'),BBANDS(close, timeperiod=10))})
        dic.update({k: v for k, v in zip(('mama2x', 'fama2x'), MAMA(close, fastlimit=0.8, slowlimit=0.1))})

        #0.5
        dic.update(
            {'dema05x': DEMA(close, timeperiod=15), 'ema05x': EMA(close, timeperiod=15),
             'ht_trendline05x': HT_TRENDLINE(close), 'kama05x': KAMA(close, timeperiod=15),
             'ma05x': MA(close, timeperiod=15),
             'midpoint05x': MIDPOINT(close, timeperiod=7), 'midprice05x': MIDPRICE(high, low, timeperiod=7),
             'sar05x': SAR(high, low, acceleration=0.01, maximum=0.1),
             'sarext05x': SAREXT(high, low, accelerationinitlong=0.01, accelerationlong=0.01,
                              accelerationmaxlong=0.1, accelerationinitshort=0.01, accelerationshort=0.01,
                              accelerationmaxshort=0.1),
             'sma05x': SMA(close, timeperiod=7), 't305x': T3(close, timeperiod=3),
             'tema05x': TEMA(close, timeperiod=15),
             'trima05x': TRIMA(close, timeperiod=15), 'wma05x': WMA(close, timeperiod=15)}

        )
        dic.update({k: v for k, v in zip(('upperband05x', 'middleband05x', 'lowerband05x'), BBANDS(close, timeperiod=3))})
        dic.update({k: v for k, v in zip(('mama05x', 'fama05x'), MAMA(close, fastlimit=0.25, slowlimit=0.025))})
        self.set(dic, dates)

class TALibMomentum(TALib):
    def __init__(self, open, high, low, close, volume, dates):
        dic = {'adx': ADX(high, low, close), 'bop': BOP(open, high, low, close), 'cci': CCI(high, low, close), 'cmo': CMO(close), 'rsi': RSI(close),
               'mfi': MFI(high, low, close, volume), 'mom': MOM(close), 'roc': ROC(close)}
        dic.update({k: v for k, v in zip(('aroondown', 'aroonup'), AROON(high, low))})
        dic.update({k: v for k, v in zip(('macd', 'macd_signal','macd_hist'), MACD(close))})
        dic.update({k: v for k, v in zip(('slow_k', 'slow_d'), STOCH(high, low, close))})
        dic.update({k: v for k, v in zip(('fast_k_rsi', 'fast_d_rsi'), STOCHRSI(close))})
        # #X2
        dic.update(
            {'adx2x': ADX(high, low, close, timeperiod=28),
             'bop2x': BOP(open, high, low, close), 'cci2x': CCI(high, low, close, timeperiod=28),
             'cmo2x': CMO(close, timeperiod=28), 'rsi2x': RSI(close, timeperiod=28),
             'mfi2x': MFI(high, low, close, volume, timeperiod=28), 'mom2x': MOM(close, timeperiod=20),
             'roc2x': ROC(close, timeperiod=20)}
        )
        dic.update({k: v for k, v in zip(('aroondown2x', 'aroonup2x'), AROON(high, low, timeperiod=28))})
        dic.update({k: v for k, v in zip(('macd2x', 'macd_signal2x','macd_hist2x'), MACD(close, fastperiod=24, slowperiod=52, signalperiod=18))})
        dic.update({k: v for k, v in zip(('slow_k2x', 'slow_d2x'), STOCH(high, low, close, fastkperiod=10, slowkperiod=6, slowdperiod=6))})
        dic.update({k: v for k, v in zip(('fast_k_rsi2x', 'fast_d_rsi2x'), STOCHRSI(close, timeperiod=28, fastkperiod=10, fastdperiod=6))})
        #
        # #X0.5
        dic.update(
            {'adx05x': ADX(high, low, close, timeperiod=7),
             'bop05x': BOP(open, high, low, close), 'cci05x': CCI(high, low, close, timeperiod=7),
             'cmo05x': CMO(close, timeperiod=7), 'rsi05x': RSI(close, timeperiod=7),
             'mfi05x': MFI(high, low, close, volume, timeperiod=28), 'mom05x': MOM(close, timeperiod=20),
             'roc05x': ROC(close, timeperiod=5)}
        )
        dic.update({k: v for k, v in zip(('aroondown05x', 'aroonup05x'), AROON(high, low, timeperiod=7))})
        dic.update({k: v for k, v in zip(('macd05x', 'macd_signal05x', 'macd_hist05x'),MACD(close, fastperiod=6, slowperiod=13, signalperiod=4))})
        dic.update({k: v for k, v in zip(('slow_k05x', 'slow_d05x'),STOCH(high, low, close, fastkperiod=4, slowkperiod=2, slowdperiod=2))})
        dic.update({k: v for k, v in zip(('fast_k_rsi05x', 'fast_d_rsi05x'),STOCHRSI(close, timeperiod=7, fastkperiod=4, fastdperiod=2))})

        self.set(dic, dates)

class TALibVolume(TALib):
    def __init__(self, high, low, close, volume, dates):

        dic = {'ad': AD(high, low, close, volume), 'adosc': ADOSC(high, low, close, volume), 'obv': OBV(close, volume)}

        #X2
        dic.update(
            {'ad2x': AD(high, low, close, volume),
             'adosc2x': ADOSC(high, low, close, volume, fastperiod=6, slowperiod=20),
             'obv2x': OBV(close, volume)}
        )

        #X0.5
        dic.update(
            {'ad05x': AD(high, low, close, volume),
             'adosc05x': ADOSC(high, low, close, volume, fastperiod=2, slowperiod=5),
             'obv05x': OBV(close, volume)}
        )

        self.set(dic, dates)

class TALibVolatility(TALib):
    def __init__(self, high, low, close, dates):
        dic = {'atr': ATR(high, low, close), 'natr': NATR(high, low, close), 'trange': TRANGE(high, low, close)}

        # 2x
        dic.update(
            {'atr2x': ATR(high, low, close, timeperiod=28),
             'natr2x': NATR(high, low, close, timeperiod=28),
             'trange2x': TRANGE(high, low, close)}
        )

        # 05x
        dic.update(
            {'atr05x': ATR(high, low, close, timeperiod=7),
             'natr05x': NATR(high, low, close, timeperiod=7),
             'trange05x': TRANGE(high, low, close)}
        )

        self.set(dic, dates)

class FFactory:
    def __init__(self, feature_dic, **kwargs):
        def ts_mean(data, period):
            assert type(data) == pd.Series
            return data.rolling(period).mean()

        def ts_rank(data, period):
            assert type(data) == pd.Series
            return data.rolling(period).apply(lambda x: pd.Series(x).rank(pct=True).values[-1])

        def ts_delta(data, period):
            assert type(data) == pd.Series
            return data.diff(period)

        def ts_std(data, period):
            assert type(data) == pd.Series
            data.rolling(period).std()

        def ts_sum(data, period):
            assert type(data) == pd.Series
            data.rolling(period).sum()

        def ts_corr(data1, data2, period):
            assert type(data1) == pd.Series
            assert type(data2) == pd.Series
            return data1.rolling(period).corr(data2)

        def ts_rate(data, period):
            assert type(data) == pd.Series
            return data.pct_change(period)

        def ts_ewm(data, com, period):
            #https://pandas.pydata.org/docs/reference/api/pandas.Series.ewm.html
            assert type(data) == pd.Series
            return data.ewm(com).mean()

        def ts_max(data, period):
            assert type(data) == pd.Series
            return data.rolling(period).max()

        def ts_min(data, period):
            assert type(data) == pd.Series
            return data.rolling(period).min()



        assert feature_dic['base']
        assert feature_dic['momentum']
        assert feature_dic['volatility']
        assert feature_dic['volume']
        assert feature_dic['formula']

        feature_dic['base'] = [f'{x}_' for x in feature_dic['base']]
        formulas = feature_dic['formula'].copy()
        del feature_dic['formula']
        features = [item for sublist in list(feature_dic.values()) for item in sublist]

        for k,v in kwargs.items():
            feat_info = v.__dict__.copy()
            if k == 'base':
                prevent_built_in_error = '_'
            else:
                prevent_built_in_error = ''
            for fkey, fval in feat_info.items():
                exec(f'{fkey}{prevent_built_in_error} = feat_info["{fkey}"].copy()')

        dataset = []
        for feature in features:
            dataset.append(eval(feature))
        dataset = pd.concat(dataset, axis=1)

        for formula in formulas:
            exec(f'formula_value = {formula}')
            dataset[formula] = formula_value.copy()

        return dataset

        # for k, obj in material.items():
        #     for fkey, fval in obj.__dict__.items():
        #         exec('')
        # for k, v in kwargs.items():
        #     temp = ''
        #     if k == 'open':
        #         temp = '_price'
        #     exec(f'{k}{temp}=self._dic["{k}"].copy()')

        # def rolling_window(a, window):
        #     shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        #     strides = a.strides + (a.strides[-1],)
        #     farr = np.zeros((window - 1, window))
        #     farr[:] = np.nan
        #     barr = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        #     return np.concatenate((farr, barr), axis=0)
        #
        # def tsrank(arr, window):
        #     from scipy.stats import rankdata
        #     arr = rankdata(rolling_window(arr, window), axis=1)
        #     return np.apply_along_axis(lambda x: x[-1] / max(x), axis=1, arr=arr)

        # def tsrank(arr, window):
        #     ts_rank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        #     return np.array(pd.Series(arr).rolling(window=window, center=False).apply(ts_rank))
        #
        # temp, self._dic = {}, kwargs.copy()
        # self.dic = {}
        #
        # for v in self._dic.values():
        #     temp.update(v.dic)
        # self._dic = temp.copy()
        # del temp
        #
        # for k, v in self._dic.items():
        #     temp = ''
        #     if k == 'open':
        #         temp = '_price'
        #     exec(f'{k}{temp}=self._dic["{k}"].copy()')
        # del k,v
        #
        # windows = [3, 5, 10, 20]
        # for window in windows:
        #     for k, v in self._dic.items():
        #         self.dic[f'[tsrank|{window}]{k}'] = tsrank(v, window)
        #
        # self.set(self.dic, dates)
        #
        # temp = [pd.Series(x) for x in self.temp_dic.values()]
        # temp_df = pd.concat(temp, axis=1).set_index(dates)




def make_dataset(feature_dic, df, tEvents, out=None):
    base = TALibBase(df.loc[:,['price', 'volume']].copy(deep=True) if 'price' in df
                             else df.loc[:, ['open', 'high', 'low', 'close','volume']].copy(deep=True), tEvents)

    open, high, low, close, volume, dates = base.open, base.high, base.low, base.close, base.volume, tEvents

    # 계산하지 말고 data 저장하고 읽어오는 방식으로 바꾸기
    # 가격 데이터 저장할 때만 게산해서 저장
    overlap = TALibOverlap(high, low, close, dates)
    momentum = TALibMomentum(open, high, low, close, volume, dates)
    volume_feature = TALibVolume(high, low, close, volume, dates)
    volatility = TALibVolatility(high, low, close, dates)
    #ts_feature = FFactory(feature_dic, base=base, overlap=overlap, momentum=momentum, volume_feature=volume_feature,volatility=volatility)
    dic = {'base': base, 'overlap': overlap, 'momentum': momentum, 'volume': volume_feature, 'volatility': volatility}
    dataset = pd.concat([dic[key].get_features(feature_dic[key]) for key in feature_dic], axis=1).dropna()
    #dataset = FFactory(feature_dic, base=base, overlap=overlap, momentum=momentum, volume_feature=volume_feature,volatility=volatility)


    if not (out is None):
        dataset = pd.concat([dataset, out.bin], axis=1).dropna()

    return base.get_features(['open', 'high', 'low', 'close', 'volume']), dataset

#
# def sep_feature(ft):
#     sep = []
#     for f in ft[:]:
#         if f == '[':
#             st, end = ft.find('['), ft.find(']')
#             sep.append(ft[st:end + 1])
#             ft = ft[end+1:]
#     sep.append(ft)
#     return sep

