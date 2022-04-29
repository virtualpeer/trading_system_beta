import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Evaluate:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.assume_worst = kwargs.get('assume_worst', False)

    def get_ror_series(self):
        if self.assume_worst:
            ret, action, fee = self.ret, self.action, self.fee
            rors = pd.Series(1., index=action.index, name='ror')
            buy, sell = action.loc[action > 0], action.loc[action < 0]
            rors.loc[buy.index] = 1 + buy * (ret.loc[buy.index] - 2 * fee)
            rors.loc[sell.index] = 1 + sell * (ret.loc[sell.index] + 2 * fee)
        else:
            ret, action = self.ret, self.action
            rors = self.ret*self.action + 1
            rors.name = 'ror'
        return rors

    def get_ror(self):
        return self.get_ror_series().product()

    def get_ideal_ror(self):
        return (self.ret*self.action + 1).product()

    def get_dd(self):
        trajectory = self.get_ror_series().cumprod()
        previous_high = trajectory.expanding().max()
        drawdown = (trajectory - previous_high) / previous_high
        return drawdown

    def get_mdd(self):
        return self.get_dd().min() * 100

    def get_accuracy(self):
        ret, action = self.ret, self.action
        from sklearn.metrics import accuracy_score
        return accuracy_score(np.sign(ret), np.sign(action))

    def get_volatility(self):
        ret = self.ret
        return ret.std()

    def get_tuw(self):
        dd = self.get_dd()
        return pd.Series(dd.loc[dd == 0].sort_index().index).diff().max()

    def get_action(self):
        return self.action

    def get_pred(self):
        return self.pred

    def get_prob(self):
        return self.prob

    def get_trade_check(self):
        trade = pd.Series()
        out, ror_series = self.out, self.get_ror_series()
        idxs = [x for x in out.index if x >= ror_series.index[0]]

        for idx in idxs:
            df0 = ror_series.loc[idx:out['tl'].loc[idx]]
            trade[idx] = df0.product()

        out = out.loc[idxs]
        out.loc[idxs, 'backtest ret'] = trade
        out.name = 'trade check'
        return out


def get_cagr(total_ret, total_day):
    return pow(total_ret+1,365/total_day)-1

def get_sharpe(daily_ror):
    assert type(daily_ror) == pd.Series
    return np.sqrt(365) * (daily_ror.mean() / daily_ror.std())


def traceinfo(trj):
    assert type(trj) == pd.Series
    trj = trj.to_frame('trj')
    trj['prev_high'] = trj['trj'].expanding().max()
    trj['drawdown'] = (trj['trj'] - trj['prev_high']) / trj['prev_high']
    return trj

def nav_info(nav):
    assert type(nav) == pd.Series
    if nav.isna().sum() > 0:
        nav.fillna(method='ffill', inplace=True)
    nav = nav.to_frame('nav')
    # init row
    nav.loc[nav.index[0] - pd.Timedelta('1d')] = {'nav': 1, 'prev_high': 1, 'drawdown': 0}
    nav.sort_index(inplace=True)
    nav['prev_high'] = nav['nav'].expanding().max()
    nav['drawdown'] = (nav['nav'] - nav['prev_high']) / nav['prev_high']
    return nav

def label_info(label, data):
    open, high, low, close, volume = data.open, data.high, data.low, data.close, data.volume
    enter, exit = close.loc[label.index].values, close.loc[label.tl].values
    label['price gap'] = abs(enter - exit)
    label['gap ratio'] = label['price gap'] / close.loc[label.index]
    return label

def plot_graph(daily_trace_df, **kwargs):
    assert set(daily_trace_df.columns).issuperset({'trj', 'prev_high', 'drawdown'})
    assert type(daily_trace_df) == pd.DataFrame

    dates, trj, bh, prev_h, dd = daily_trace_df.index, daily_trace_df.trj, daily_trace_df.bh, daily_trace_df.trj, daily_trace_df.drawdown

    ror = trj.pct_change(1).fillna(0).values
    rolling_sharpe = trj.pct_change(1).rolling('90d').apply(get_sharpe).values
    rolling_mdd = dd.rolling('90d').min().values

    fig = plt.figure(figsize=(16, 20))
    ax1, ax2, ax3, ax4, ax5 = fig.add_subplot(5, 1, 1), fig.add_subplot(5, 1, 2), fig.add_subplot(5, 1, 3), fig.add_subplot(5, 1, 4), fig.add_subplot(5, 1, 5)
    ax1.grid(); ax2.grid(); ax3.grid();

    #NAV
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.plot(dates, trj, color='firebrick')
    ax1.plot(dates, bh, color='plum')
    ax1.set_xlabel('Date'); ax1.set_ylabel('nav');
    ax1.set_title(f'{kwargs["symbol"]} - Trajectory')

    #DD
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.plot(dates, dd*100, color='royalblue')
    ax2.set_xlabel('Date'); ax2.set_ylabel('dd (%)');
    ax2.set_title(f'Drawdown')

    #ROR
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.bar(dates, ror*100, color='green')
    ax3.set_xlabel('Date'); ax3.set_ylabel('return (%)');
    ax3.set_title(f'Daily Return')

    #rolling sr
    ax4.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax4.plot(dates, rolling_sharpe, color='orange')
    ax4.set_xlabel('Date'); ax4.set_ylabel('sharpe(90d)');
    ax4.set_title(f'Rolling Sharpe')

    #rolling mdd
    ax5.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax5.plot(dates, rolling_mdd*100, color='cornflowerblue')
    ax5.set_xlabel('Date'); ax5.set_ylabel('mdd(%, 90d)');
    ax5.set_title(f'Rolling MDD')

    #Save
    plt.gcf().autofmt_xdate()
    save_time = pd.Timestamp.today().strftime('%Y-%m-%d_%H%M%S')
    plt.savefig(f'{kwargs["path"]}\\{kwargs["symbol"]}_plot_graph_{save_time}.png', bbox_inches='tight')
    plt.close()