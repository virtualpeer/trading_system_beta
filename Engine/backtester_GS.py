import multiprocessing as mp
import os, json
from datetime import datetime
from Engine.Core.World import World
import Config.feature.feature_search_config as ftsh
import itertools as it
from tqdm import tqdm

def feature_analysis(trader_dic, path):
    world = World(config_json=trader_dic)
    world.analysis(path=path)

def simulation(trader_dic, path):
    world = World(config_json=trader_dic)
    world.save(path=path)

def mk_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    groups = it.zip_longest(*args, fillvalue=fillvalue)
    groups = [list(x) for x in list(groups)]
    new_groups = []
    for group in groups:
        new_group = [x for x in group if x != None]
        new_groups.append(new_group)
    return new_groups

if __name__ == '__main__':
    path_list = os.getcwd().split(os.path.sep)
    path_list = path_list[:path_list.index('trading_system_beta') + 1]
    config_path = os.path.join(os.path.sep.join(path_list), 'Config')
    config_exchange_path = os.path.join(config_path, 'exchange')
    config_trader_path = os.path.join(config_path, 'trader')
    simulation_grid_search = True  # False

    # trader dic
    with open(os.path.join(config_trader_path,'trader_data_type.json'), 'r', encoding='UTF8') as read_file:
        trader_dic = json.load(read_file)
    trader_dic.update(ftsh.default_config_params)

    if simulation_grid_search:
        save_path = os.path.join(os.getcwd(), 'backtest_result')
        save_path = os.path.join(save_path, f'backtester_GS {datetime.now().strftime("%Y-%m-%d %H-%M-%S")}')
        target = simulation
    else:
        save_path = os.path.join(os.getcwd(), 'feature_anlysis')
        save_path = os.path.join(save_path, f'{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}')
        target = feature_analysis
    mk_folder(save_path)


    procs = []
    combinations = it.product(*(ftsh.search_config_params[k] for k in ftsh.search_config_params))
    combinations = list(combinations)[:]
    for comb in combinations[:1]:
        change_dic = {k:v for k,v in zip(ftsh.search_config_params.keys(), comb)}
        file_element = []
        for k, v in change_dic.items():
            for rep in ['.', ',', ':', ' ', '/USDT', '\'', '[', ']', '{', '}']:
                v = str(v).replace(rep, '')
            file_element.append(ftsh.short_file_name[k]+v.replace('hours','H'))
        change_dic['param name'] = '_'.join(file_element)
        change_dic['file name'] = change_dic['symbol'].replace('/','-')
        trader_dic.update(change_dic)
        proc = mp.Process(target=target, kwargs={'trader_dic': trader_dic.copy(), 'path': save_path})
        procs.append(proc)

    #CPU, RAM 상황에 맞춰서 숫자 정하기 : N개 프로세스 동시 처리
    group_procs = grouper(1, procs)
    for num, group_proc in tqdm(enumerate(group_procs)):
        for proc in group_proc:
            proc.start()

        for proc in group_proc:
            proc.join()

        for proc in group_proc:
            proc.terminate()

        print(f'{num} group simulation completed !!')
