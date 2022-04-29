
import Config.feature.feature_store as ftst

symbols = ['BTC', 'ETH']

search_config_params = {
    'symbol' : [f'{x}/USDT' for x in symbols], #7867
    'ptSl' : [  [0.7, 0.7],
                [1.0, 1.0]],
    'event frequency' : [{'hours' : 24},
                         {'hours' : 48},
                         {'hours' : 72},
                         {'hours' : 96}],
    'vertical barrier' : [{'hours' : 1},
                          {'hours' : 12},
                          {'hours' : 24},
                          {'hours' : 48}],
    'minRet' : [0.025, 0.03, 0.035],
    'event cutoff coefficient' : [2.0, 2.5],
    'unit bars per hour' : [6, 18],
    'amplify' : [1.1],
    'first days for unit' : [{"days": 180}],
    'first days for volatility' : [{"days": 180}],
    'model_type' : [1]
}


default_config_params = {
    'features' : ftst.btc_tech_heavy,
    'amplify' : 1.1,
    'warmup' : 0.2,
    'backtesting method': "WF",
    'transaction fee': 0.0015,
    'save analysis' : False,
    'save plot' : False,
    'strategy' : 'average'
}

short_file_name = {
    'ptSl' : 'ps',
    'symbol' : 'sym',
    'amplify' : 'amp',
    'event frequency' : 'efrq',
    'vertical barrier' : 'vbar',
    'minRet' : 'mRt',
    'event cutoff coefficient' : 'ecof',
    'features' : 'feat',
    'warmup' : 'wmp',
    'backtesting method' : 'meth',
    'transaction fee' : 'fee',
    'unit bars per hour' : 'ubph',
    'first days for unit' : 'fdfu',
    'first days for volatility' : 'fdfv',
    'model_type' : 'mdt'
}