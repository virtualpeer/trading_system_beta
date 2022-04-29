

base_feature = {
        'base': ['volume', 'close', 'open', 'high', 'low'],
        'momentum': ['adx', 'bop', 'cci', 'cmo', 'rsi', 'mfi', 'mom', 'roc', 'aroondown', 'aroonup', 'macd', 'macd_signal',
                      'macd_hist', 'slow_k', 'slow_d', 'fast_k_rsi', 'fast_d_rsi', 'adx2x', 'bop2x', 'cci2x', 'cmo2x', 'rsi2x',
                      'mfi2x', 'mom2x', 'roc2x', 'aroondown2x', 'aroonup2x', 'macd2x', 'macd_signal2x', 'macd_hist2x', 'slow_k2x',
                      'slow_d2x', 'fast_k_rsi2x', 'fast_d_rsi2x', 'adx05x', 'bop05x', 'cci05x', 'cmo05x', 'rsi05x', 'mfi05x',
                      'mom05x', 'roc05x', 'aroondown05x', 'aroonup05x', 'macd05x', 'macd_signal05x', 'macd_hist05x', 'slow_k05x',
                      'slow_d05x', 'fast_k_rsi05x', 'fast_d_rsi05x'],
        'volume': ['adosc', 'adosc2x', 'adosc05x', 'ad', 'ad2x', 'ad05x', 'obv', 'obv2x', 'obv05x'],
        'volatility': ['atr', 'atr2x', 'atr05x', 'natr', 'natr2x', 'natr05x', 'trange', 'trange2x', 'trange05x'],
        'overlap' : ['dema', 'ema', 'ht_trendline', 'kama', 'ma', 'midpoint', 'midprice', 'sar', 'sarext',
                      'sma', 't3', 'tema', 'trima', 'wma', 'upperband', 'middleband', 'lowerband', 'mama', 'fama',
                      'dema2x', 'ema2x', 'ht_trendline2x', 'kama2x', 'ma2x', 'midpoint2x', 'midprice2x', 'sar2x', 'sarext2x',
                      'sma2x', 't32x', 'tema2x', 'trima2x', 'wma2x', 'upperband2x', 'middleband2x', 'lowerband2x', 'mama2x',
                      'fama2x', 'dema05x', 'ema05x', 'ht_trendline05x', 'kama05x', 'ma05x', 'midpoint05x', 'midprice05x', 'sar05x',
                      'sarext05x', 'sma05x', 't305x', 'tema05x', 'trima05x', 'wma05x', 'upperband05x', 'middleband05x', 'lowerband05x',
                      'mama05x', 'fama05x']}


btc_tech_heavy = {'base': ['close', 'low', 'high', 'open', 'volume'],
                'momentum': ['cci2x', 'roc2x', 'slow_d', 'macd2x', 'roc', 'slow_d05x',
                            'slow_d2x', 'mfi2x', 'mfi05x', 'mfi', 'adx05x', 'adx'],
                'volume': ['obv2x', 'obv', 'obv05x', 'adosc05x', 'ad2x', 'ad05x', 'ad',
                            'adosc', 'adosc2x'],
                'volatility': ['natr', 'trange2x', 'trange05x', 'trange',
                                'natr05x', 'natr2x', 'atr05x', 'atr2x', 'atr'],
                #'formula' : []
                }


btc_tech_mid = {'base': ['low', 'high', 'open', 'volume'],
                'momentum': ['mfi2x', 'mfi05x', 'mfi', 'adx05x', 'adx'],
                'volume': ['obv05x', 'adosc05x', 'ad2x', 'ad05x', 'ad',
                            'adosc', 'adosc2x'],
                'volatility': ['natr', 'trange2x', 'trange05x', 'trange', 'natr05x',
                            'natr2x', 'atr05x', 'atr2x', 'atr']}


btc_tech_light = {'base': ['low', 'high', 'open', 'volume'],
 'momentum': ['adx05x', 'adx'],
 'volume': ['ad2x', 'ad05x', 'ad', 'adosc', 'adosc2x'],
 'volatility': ['natr2x', 'atr05x', 'atr2x', 'atr']}