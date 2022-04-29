import pandas as pd

from Tools.Feature import *
from Tools.Analysis import *
from Preprocessor.CUMSUMPreprocessor import CUMSUMPreprocessor
from Label.TripleBarrier import TripleBarrier
from Model.Bagging import Bagging
from Strategy.Strategy import Strategy
from Strategy.SimpleStrategy import SimpleStrategy
from Strategy.AverageStrategy import AverageStrategy
import os, json, pickle
from functools import reduce
from Tools.Evaluate import *

class World:

    def __init__(self, config_json):
        self._config = config_json
        # need to write parameters parse_dates=['dates']
        # need to call df.set_index('dates', inplace=True)
        # expect the data to look like columns include ['open', 'high', 'low', 'close','volume'] or ['price', 'volume'],
        # index_name='dates'

        #[DEBUG] 거래소 데이터와 같음을 확인
        self._data = pd.read_csv(os.path.join(self._config['file path'], self._config['file name']+'.csv'),
                                 parse_dates=['dates']).set_index('dates')

        assert not (True in self._data.index.duplicated())

        # [DEBUG] 거래소 데이터를 한 칸 밀어서 index가 곧 해당 close를 알게 될 수 있는 시점
        if 'close' in self._data:
            # if rawData is minute data, then the 'close' price must be the price of the instrument
            # at index time to prevent cheating!
            self._data.index = pd.Index(np.append(self._data.index.values[1:],
                                                  self._data.index.values[-1] + np.timedelta64(1, 'm')),
                                        name='dates')

        unit_bars_per_day = self._config['unit bars per hour'] * self._config['hours per day']
        relevant = self._data.loc[:self._data.index[0]+pd.Timedelta(**self._config['first days for unit']), self._config['mode']]
        self._config.update({'unit': relevant.sum() / (unit_bars_per_day*np.unique(relevant.index.date).shape[0])})

        preprocessor = CUMSUMPreprocessor(target_df=self._data, preprocessor_config=self._config)
        bar_df, self._residue_unit, (unit, std) = preprocessor.make_bar_df()
        print(f'unit and std in bar df: \n {unit, std}')

        #[DEBUG] data_price == self._data
        self._data_price = self._data.price.copy(deep=True) if 'price' in self._data else self._data.close.copy(deep=True)
        #[DEBUG] bar_df == self._bar_price
        self._bar_price = bar_df.price.copy(deep=True) if 'price' in bar_df else bar_df.close.copy(deep=True)
        self.bar_vol = get_vol(self._bar_price, **self._config['event frequency'])

        self.mean_vol = self.bar_vol.loc[:self.bar_vol.index[0]+pd.Timedelta(**self._config['first days for volatility'])].mean()

        tEvents, self._residue_sPos, self._residue_sNeg \
            = preprocessor.getTEvents(df=np.log(self._bar_price),
                                      upper=np.log(1 + self._config['event cutoff coefficient']*self.mean_vol),
                                      lower=np.log(1 - self._config['event cutoff coefficient']*self.mean_vol))
        print(f'PtSl width statistics: \n {(self.bar_vol*self._config["ptSl"][0]).describe()} \n')

        print(f'the number of events in each day: \n {get_frequency(self._data, tEvents)} \n'
              f'the number of unit bars in each day: \n {get_frequency(self._data, bar_df.index)}')

        label = TripleBarrier(price=self._data_price, tEvents=tEvents, label_config=self._config)
        out = label.getBins(trgt=self.bar_vol)

        self._base, self._dataset = make_dataset(self._config['features'], self._data, tEvents, out)
        self._out = out.loc[self._dataset.index]
        self.model_generator = \
            Bagging(events=self._out[['tl']].copy(deep=True), price=self._data_price,
                    numThreads=self._config['numThreads'], clfLastW=self._config['clfLastW'])
        self._model = self.model_generator.model(type=self._config['model_type'],
                                                 n_estimators=self._config['n estimators'],
                                                 max_samples=self.model_generator.sampleTw.mean(), oob_score=True)

        print(f'average uniqueness score per data point: \n {self.model_generator.sampleTw.describe()}')
        print(f'sample weight per data point: \n {self.model_generator.sample_weight.describe()}')
        print(f'dataset: \n {self._dataset} \n model: \n {self._model} \n out: \n {self._out}')

        assert not (False in self.model_generator.sample_weight.index == self._dataset.index == self._out.tl.index)

        if self._config['backtesting method'] == 'CV':
            from Engine.Core.Backtest import PurgedKFold
            self._cvGen = PurgedKFold(tl=self._out['tl'], cv_config=self._config)
        else:
            from Engine.Core.Backtest import PurgedWF
            self._cvGen = PurgedWF(tl=self._out['tl'], cv_config=self._config)

    def run(self):
        self._fit = []
        X, y = self._dataset.iloc[:, :-1], self._dataset.iloc[:, -1]
        for train, _ in tqdm(list(self._cvGen.split(X))):
            from sklearn.base import clone
            clf = clone(self._model)
            train_input, train_label, train_weight = \
                X.iloc[train], y.iloc[train], self.model_generator.sample_weight.iloc[train].values
            fit = clf.fit(train_input, train_label, sample_weight=train_weight)
            self._fit.append(fit)
        print('model fitted!')

    def eval(self):
        self.run()
        cvscore = []
        strategy = self._config['strategy']
        strategyscores = {strategy : {metric : [] for metric in self._config['strategyScore']}}
        strategyscores.update({'buy hold': [], 'cv score': []})
        strategyscore_dfs = {strategy : {metric : [] for metric in self._config['strategyScore_df']}}

        X, y = self._dataset.iloc[:, :-1], self._dataset.iloc[:, -1]
        for fit, (_, test) in tqdm(list(zip(self._fit, self._cvGen.split(X)))):
            print(f'classifier info: {fit.n_classes_}, {fit.classes_}')
            test_input, test_label, test_weight = \
                X.iloc[test], y.iloc[test], self.model_generator.sample_weight.iloc[test].values
            prob, pred = fit.predict_proba(test_input), fit.predict(test_input)
            prob_, pred_ = pd.Series([max(probs) for probs in prob], index=test_input.index), \
                           pd.Series(pred, index=test_input.index)
            out_, price_ = self._out.loc[:test_input.index[-1]], self._data_price.loc[:test_input.index[-1]]
            general = Strategy(pred=pred_, prob=prob_, out=out_, price=price_, config=self._config)
            simple = SimpleStrategy(pred=pred_, prob=prob_, out=out_, price=price_, config=self._config)
            average = AverageStrategy(pred=pred_, prob=prob_, out=out_, price=price_, config=self._config, numClasses=fit.n_classes_)
            strategy_dic = {'general': general, 'simple': simple, 'average': average}
            out_eval = strategy_dic[strategy].evaluate()
            for idx, metric in enumerate(self._config['strategyScore'] + self._config['strategyScore_df']):
                try:
                    strategyscores[strategy][metric].append(out_eval[idx])
                except:
                    strategyscore_dfs[strategy][metric].append(out_eval[idx])

            buyhold_ = self._data_price.loc[test_input.index.max()]/ self._data_price.loc[test_input.index.min()]
            strategyscores['buy hold'].append(buyhold_)
            cvscore_ = self.__cvscore(fit, test_label, prob, pred, test_weight, self._config['cvScore'])
            strategyscores['cv score'].append(cvscore_)

        print('backtest completed!')

        for metric_df in self._config['strategyScore_df']:
            strategyscore_dfs[strategy][metric_df] = reduce(lambda x, y: x.append(y), strategyscore_dfs[strategy][metric_df]) if strategyscore_dfs[strategy][metric_df] else pd.Series(dtype=float)

        return strategyscores, strategyscore_dfs


    def analysis(self, path):
        #version 20210920
        os.mkdir(path)
        temp_model_generator = Bagging(self._out[['tl']].copy(deep=True), self._data_price,
                                       self._config['numThreads'], self._config['clfLastW'])
        temp_model = temp_model_generator.model(type=self._config['model_type'],
                                                n_estimators=self._config['n estimators'],
                                                max_samples=temp_model_generator.sampleTw.mean(),
                                                oob_score=True, base_max_features=1)

        X, y = self._dataset.iloc[:, :-1], self._dataset.iloc[:, -1]
        imp_MDI, imp_MDA, imp_SFI, pca, cvscore = \
            featImp(temp_model, X, y, temp_model_generator.sample_weight, self._config['cvScore'],
                    self._cvGen, sfi=self._config['sfi'])
        featINFO = {'MDI': imp_MDI['mean'], 'MDA': imp_MDA['mean'], 'SFI': imp_SFI['mean']}
        comb = pd.concat(featINFO, axis=1)
        comb_feat = comb.sub(comb.mean(), axis=1).div(comb.std(), axis=1).mean(axis=1).sort_values()
        featINFO['norm'], featINFO['config'] = comb_feat, self._config
        rank = pd.Series(range(1, featINFO['norm'].shape[0] + 1), index=featINFO['norm'].index[::-1])
        pca.sort_values(inplace=True)
        featINFO['norm rank'], featINFO['pca'] = rank, pca
        with open(os.path.join(path,f'{self._config["param name"]}.pkl', 'wb')) as handle:
            pickle.dump(featINFO, handle, protocol=pickle.HIGHEST_PROTOCOL)


        from scipy.stats import weightedtau
        print(f'summary : \n{featINFO["norm"]}')
        print(f'ranking of feature importance by sum of normalized MDI, MDA, SFI: \n{rank}')
        print(f'ranking of eigen values: \n{pca}')
        print(f'weighted Kendall\'s tau: \n {weightedtau(featINFO["norm"].values, pca.values**-1.0)[0]}')

        return featINFO["MDI"], featINFO["MDA"], featINFO["SFI"], featINFO["norm"], pca, cvscore

    def save(self, path, mode=False):
        strategyscores, strategyscore_dfs = self.eval()
        symbol_path = os.path.join(path, f'{self._config["file name"]}')
        if not os.path.exists(symbol_path):
            os.makedirs(symbol_path)

        data, strategy = self._data, self._config['strategy']
        split_summary = pd.DataFrame(strategyscores[strategy])
        del strategyscores[strategy]
        for k,v in strategyscores.items():
            split_summary[k] = v

        ror, bh, trade_check = strategyscore_dfs[strategy]['ror series'], strategyscore_dfs[strategy]['buy hold series'], strategyscore_dfs[strategy]['trade check']
        nav = nav_info(nav=ror.cumprod())
        d_nav = nav_info(nav=ror.cumprod().resample('1D').first())
        label = label_info(label = self._out.copy(), data=data)


        save_information = {
            'config' : pd.DataFrame(data=[str(self._config)]),
            'nav'  : nav,
            'dnav' : d_nav,
            'label' : label,
            'trade' : trade_check,
            'summary' : split_summary
        }

        if mode:
            return save_information
        else:
            writer = pd.ExcelWriter(os.path.join(symbol_path, f'{self._config["param name"]}.xlsx'), engine='xlsxwriter')
            for k, v in save_information.items():
                v.to_excel(writer, sheet_name=k)
            writer.save()




    def now(self, look_back_days):
        # return dnav
        X, y = self._dataset.iloc[:, :-1], self._dataset.iloc[:, -1]
        idx = pd.Timestamp.now() - pd.Timedelta(days=look_back_days) #hyper parameter

        from sklearn.base import clone
        clf = clone(self._model)
        train_input, train_label, train_weight = \
            X.loc[:idx], y.loc[:idx], self.model_generator.sample_weight.loc[:idx].values
        test_input, test_label, test_weight = \
            X.loc[idx:], y.loc[idx:], self.model_generator.sample_weight.loc[idx:].values

        fit = clf.fit(train_input, train_label, sample_weight=train_weight)

        strategy = self._config['strategy']
        strategyscores = {strategy: {metric: [] for metric in self._config['strategyScore']}}
        strategyscores.update({'buy hold': [], 'cv score': []})
        strategyscore_dfs = {strategy: {metric: [] for metric in self._config['strategyScore_df']}}

        prob, pred = fit.predict_proba(test_input), fit.predict(test_input)
        prob_, pred_ = pd.Series([max(probs) for probs in prob], index=test_input.index), \
                       pd.Series(pred, index=test_input.index)
        out_, price_ = self._out.loc[:test_input.index[-1]], self._data_price.loc[:test_input.index[-1]]
        general = Strategy(pred=pred_, prob=prob_, out=out_, price=price_, config=self._config)
        simple = SimpleStrategy(pred=pred_, prob=prob_, out=out_, price=price_, config=self._config)
        average = AverageStrategy(pred=pred_, prob=prob_, out=out_, price=price_, config=self._config,
                                  numClasses=fit.n_classes_)
        strategy_dic = {'general': general, 'simple': simple, 'average': average}
        out_eval = strategy_dic[strategy].evaluate()

        for idx, metric in enumerate(self._config['strategyScore'] + self._config['strategyScore_df']):
            try:
                strategyscores[strategy][metric].append(out_eval[idx])
            except:
                strategyscore_dfs[strategy][metric].append(out_eval[idx])

        return strategyscores, strategyscore_dfs


    @staticmethod
    def __cvscore(fit, y, prob, pred, sample_weight, scoring):
        if not (scoring in ['neg_log_loss', 'accuracy']):
            raise Exception('wrong cv scoring method.')
        from sklearn.metrics import log_loss, accuracy_score
        if scoring == 'neg_log_loss':
            score_ = -log_loss(y, prob, sample_weight=sample_weight, labels=fit.classes_)
        else:
            score_ = accuracy_score(y, pred, sample_weight=sample_weight)
        return score_
