import os, sqlite3
import pandas as pd
from datetime import datetime, timedelta
from itertools import chain
from tqdm import tqdm
import glob
import numpy as np
import time
import multiprocessing as mp


def fix_duplicate(df, format='%Y-%m-%d %H:%M:%S', unit='ms'):
    dt = pd.to_datetime(df.index, format=format)
    delta = pd.to_timedelta(df.groupby(dt).cumcount(), unit=unit)
    df.index = dt + delta.values


def db_to_csv(db_path: str, data_path: str, code: str or int):
    """
    :param db_path: directory where stock_price.db is placed
    :param code: the stock code you wish to call (it must have been already called
    :return: None
    """
    cnx = sqlite3.connect(os.path.join(db_path, 'stock_price.db'))
    stock_code = f'\"{code}\"'
    data = pd.read_sql_query(f'SELECT * FROM {stock_code}', cnx)
    data['index'] = pd.to_datetime(data['index'])
    path = os.path.join(data_path, 'stock_price')
    if not os.path.exists(path):
        os.makedirs(path)
    data.to_csv(os.path.join(path, f'{code}.csv'), index=False)


def minute_to_csv(txt_path, file_name, date_format='%m/%d/%Y%H:%M'):
    columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    a = pd.read_csv(os.path.join(txt_path, file_name), header=None).rename(
        columns={i: column for i, column in enumerate(columns)}) \
        .assign(dates=lambda x: pd.to_datetime(x.date + x.time, format=date_format)) \
        .set_index('dates').drop_duplicates() \
        .drop(columns=['date', 'time'])
    assert not (True in a.index.duplicated())
    # fix_duplicate(a, unit='S')
    a.to_csv(os.path.join(txt_path, file_name[:file_name.rfind('txt')] + 'csv'))


def tick_to_csv(txt_path, file_name, date_format='%m/%d/%Y%H:%M:%S'):
    columns = ['date', 'time', 'price', 'bid', 'ask', 'size']
    a = pd.read_csv(os.path.join(txt_path, file_name), header=None).rename(
        columns={i: column for i, column in enumerate(columns)}) \
        .assign(dates=lambda x: pd.to_datetime(x.date + x.time, format=date_format)) \
        .assign(volume=lambda x: x['size']) \
        .assign(dollar=lambda x: x['size'] * x.price) \
        .set_index('dates').drop_duplicates() \
        .drop(columns=['date', 'time', 'size'])
    if True in a.index.duplicated():
        fix_duplicate(a)
    assert not (True in a.index.duplicated())
    a.to_csv(os.path.join(txt_path, file_name[:file_name.rfind('txt')] + 'csv'))


def make_new_rawData(exchange, symbol, path, rawData):
    # rawData: price of an instrument at index time == 'open' column
    rawData_ = rawData.iloc[:-1]

    def crawl():
        since = None
        while True:
            a = exchange.fetch_ohlcv(symbol=symbol, timeframe='1m', since=since, limit=1500)
            try:
                latest_datetime = datetime.fromtimestamp(a[-1][0] / 1000)
            except IndexError:
                break

            if latest_datetime <= rawData_.index[-1]:
                break
            else:
                since = int(datetime.timestamp(datetime.fromtimestamp(a[0][0] / 1000)
                                               - timedelta(minutes=1500)) * 1000)
                yield a
            time.sleep(0.01)

    c = pd.DataFrame(chain.from_iterable(crawl()))
    c['dates'] = c.apply(func=lambda row: datetime.fromtimestamp(row[0] / 1000), axis=1)
    c = c.drop(columns=[0]).rename(columns={1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}) \
            .set_index('dates').append(rawData_).sort_index().drop_duplicates().iloc[:-1]
    file_save_path = path
    if not os.path.exists(file_save_path):
        os.makedirs(file_save_path)
    c.to_csv(os.path.join(file_save_path, symbol.replace('/', '-') + '.csv'))


def make_new_rawData_for_all(exchange, dic):
    file_path = dic['file path']
    # multiprocess 빠르게 처리;

    procs = []
    for file in tqdm(os.listdir(file_path)):
        file_name = file[:file.rfind('.')]
        symbol = file_name.replace('-', '/')
        rawData = pd.read_csv(os.path.join(file_path, file),
                              parse_dates=['dates']).set_index('dates')
        make_new_rawData(exchange=exchange, symbol=symbol, path=file_path,
                         rawData=rawData)


if __name__ == "__main__":
    import json, ccxt
    path_list = os.getcwd().split(os.path.sep)
    path_list = path_list[:path_list.index('ml_assetallocation') + 1]
    config_path = os.path.join(os.path.sep.join(path_list), 'Config')
    config_exchange_path = os.path.join(config_path, 'exchange')
    config_trader_path = os.path.join(config_path, 'trader')

    with open(f'{config_exchange_path}/exchange_data.json', 'r', encoding='UTF8') as read_file:
        exchange_dic = json.load(read_file)

    with open(f'{config_trader_path}/trader_data_type.json', 'r', encoding='UTF8') as read_file:
        trader_dic = json.load(read_file)

    file_paths = glob.glob(trader_dic['file path'] + '/*.csv')
    for x in tqdm(file_paths):
        symbol = x.split('\\')[-1].split('.')[0].replace('-', '/')
        exchange = ccxt.binance(exchange_dic)
        rawData = pd.read_csv(x, parse_dates=['dates']).set_index('dates')
        data_valid1 = (type(rawData.index[0]) == pd.Timestamp)  # date에 이상한 값 있는 경우 parse_dates가 안 먹히는 경우 발생
        data_valid2 = len(rawData.index) == len(
            pd.date_range(rawData.index[0], rawData.index[-1], freq='1T'))  # 중간에 데이터 빠진 거 있는 경우
        temp = set(pd.date_range(rawData.index[0], rawData.index[-1], freq='1T')) - set(rawData.index)
        if not (data_valid1):
            rawData.iloc[:2].to_csv(x)
            rawData = pd.read_csv(x, parse_dates=['dates']).set_index('dates')
            print(f'[DATA BUG] {x}, [data_valid1] : {data_valid1}, [data_valid2] : {data_valid2}')
            assert len(rawData.index) == len(pd.date_range(rawData.index[0], rawData.index[-1], freq='1T'))

        make_new_rawData(exchange=exchange, symbol=symbol, path=trader_dic['file path'], rawData=rawData)
