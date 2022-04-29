import os
import sqlite3
import pandas as pd


def fix_duplicate(df, format='%Y-%m-%d %H:%M:%S', unit='ms'):
    dt = pd.to_datetime(df.index, format=format)
    delta = pd.to_timedelta(df.groupby(dt).cumcount(), unit=unit)
    df.index = dt + delta.values

def db_to_csv(db_path: str, code: str or int):
    """
    :param db_path: directory where stock_price.db is placed
    :param code: the stock code you wish to call (it must have been already called
    :return: None
    """
    cnx = sqlite3.connect(db_path+'stock_price.db')
    stock_code = f'\"{code}\"'
    data = pd.read_sql_query(f'SELECT * FROM {stock_code}', cnx)
    data['index'] = pd.to_datetime(data['index'])
    path = os.path.join(os.getcwd(), 'stock_price')
    if not os.path.exists(path):
        os.mkdir(path)
    data.to_csv(os.path.join(path, f'{code}.csv'), index=False)

def txt_to_csv(txt_path: str):
    a = pd.read_csv(os.path.join(txt_path, 'IVE_tickbidask.txt'), header=None).rename(columns={0: 'date', 1: 'time', 2: 'price', 3: 'bid', 4: 'ask', 5: 'size'}) \
        .assign(dates=lambda x: pd.to_datetime(x.date + x.time, format='%m/%d/%Y%H:%M:%S')) \
        .assign(volume=lambda x: x['size']) \
        .assign(dollar=lambda x: x['size'] * x.price) \
        .set_index('dates').drop_duplicates()\
        .drop(columns=['date', 'time'])
    if True in a.index.duplicated():
        fix_duplicate(a)
    assert not (True in a.index.duplicated())
    a.to_csv(os.path.join(txt_path, 'S&P_data.csv'))