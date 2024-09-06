import datetime
import glob
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import string
import seaborn as sns
import sys
sys.path.insert(1,'/home/axelbm23/Code/ML_AI/Algos/ReinforcementLearning/')
from agents import DDQN, DDQN_tradexecution, TWAP, RANDOM_TE
from environments import TradeExecution 
import tensorflow as tf
from typing import Any, Tuple, Optional


def write_pkl(data:Any, filename:str, path:str) -> None:
    """
    Writes data object as a pkl object in path
    """
    with open(f'{path}/{filename}.pkl', 'r') as f:
        pickle.dump(data, f)

def load_pkl(filename:str, path:str) -> Any:
    """
    Loads pickle file in path/filename
    """
    return pickle.load(f'{path}/{filename}.pkl')

def load_nparray(filename:str, path:str) -> np.array:
    """
    Loads numpy array from path/filename
    """
    return np.load(f'{path}/{filename}')

def write_nparray(arr:np.array, filename:str, path:str) -> None:
    """
    Saves numpy arrray arr into path folder as a binary object
    """
    np.save(f'{path}/{filename}', arr)

def get_ids(path:str) -> dict:
    """
    Given a txt file with two identifiers
    separated with comma, this function
    returns a dict with the first element
    as key and the second element as value
    """
    ids = {}
    filename = f'{path}/codes.txt'
    if not os.path.exists(filename):
        return ids
        
    with open(filename,'r') as f:
        sett_id, id = f.readline()
        ids[sett_id] = id
    
    return ids

# Check if experiment has already been run
def is_exp_run(id:str, mode:str, path:str) -> Optional[str]:
    """
    Returns True if given agent has been run on the mode=mode
    with settings=sett
    """

    # Read the file containing the map between the settings and
    # unique identifiers
    if mode=='train':
        ids = get_ids(path)
        if not ids or id not in ids:
            return None
        
        if id in ids:
            return ids[id]
    
    elif mode=='test':
        # TO-DO: Implement part of mode=test
        print('Still to do!')
    
    else:
        raise ValueError(f'mode={mode} is not supported, check input!')
    

def update_sett(cm_sett:dict, new_key:str, new_val:Any, agent:str) -> dict:
    """
    Updates the common settings with the new value, new val, 
    specified as new_key and returns a new dictionary
    """

    new_dict = cm_sett.copy()
    
    new_dict['pretrain'] = agent == 'ddqn_pre'

    if new_key=='default':
        return new_dict
    
    if new_key in cm_sett:
        new_dict[new_key] = new_val
        return new_dict
    
    if new_key=='target_copy':
        new_dict['nn_cpy_cad'] = new_val[0]
        new_dict['soft_update'] = new_val[1]
        return new_dict

def create_trial_id(agent:str, params:dict, **kwargs) -> str:
    """
    Given a params dictionary, returns a string concatenating key_value-
    """
    id = [agent]
    if agent=='twap':
        id.append(f'epi_{params['epi']}')
    else:
        iter = kwargs.get('iter', None)
        id.append(f'iter_{iter}')
        for key,value in params.items():
            if key!='n_iter':
                id.append(f'{key}_{value}')
    
    return '-'.join(id)

def aggregate_data(df: pd.DataFrame) -> pd.DataFrame:

    # Generate equally sampled intervals of granularity
    # from min_dt and max_dt
    df['datetime(s)'] = df['datetime'].apply(
        lambda x: x.replace(microsecond=0))
    # Compute min and max
    min_dt = df['datetime(s)'].min()
    max_dt = df['datetime(s)'].max()
    df = df.groupby(['datetime(s)']).last().reset_index()[
        ['datetime(s)', 'mp']]
    new_df = pd.DataFrame(pd.date_range(
        start=min_dt, end=max_dt, freq='1s'), columns=['datetime(s)'])
    new_df = pd.merge(new_df, df, left_on='datetime(s)',
                      right_on='datetime(s)')

    # if there's any nan in df, fill it with the last value
    new_df.ffill(inplace=True)
    return new_df

def load_btcusd(path: str,  output_csv: str):
    if not os.path.exists(output_csv):
        files = glob.glob(path)
        entire_data = []
        for file in files:
            df = pd.read_csv(file)
            df['datetime'] = df['transaction_time'].apply(
                lambda x: datetime.datetime.fromtimestamp(x/1_000))
            df['mp'] = 0.5*(df['best_bid_price']+df['best_ask_price'])
            df = aggregate_data(df[['datetime', 'mp']])
            entire_data.append(df)

        # Remove duplicates if any
        df = pd.concat(entire_data)
        # Finally compute the diff between mid prices (will speed calculations
        # later on)
        df['mp_diff'] = -1*df['mp'].diff(-1)
        df.drop_duplicates(inplace=True)
        df.reset_index(inplace=True, drop=True)
        df.sort_values('datetime(s)', inplace=True)
        df.to_csv(output_csv)
    else:
        df = pd.read_csv(output_csv)
    return df

def format_history(hist: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(hist, columns=['qt', 't', 'action', 'reward', 'algo', 'df_idx', 'episode'])

def format_logs(data: np.array) -> pd.DataFrame:
    df = pd.DataFrame(
        data, columns=['state', 'action', 'reward', 'state_prime'])
    df[['qt', 't']] = df['state'].apply(
        lambda x: pd.Series(x[0]))
    df[['qt_prime', 't_prime']] = df['state_prime'].apply(
        lambda x: pd.Series(x[0]))

    df.drop(['state', 'state_prime'], axis=1, inplace=True)

    return df[['qt', 't', 'action', 'reward', 'qt_prime', 't_prime']]

def plot(data: list, title: str, mavg: bool) -> None:
    serie = pd.Series(data)
    if mavg:
        mavg = serie.rolling(window=15).mean()
    plt.plot(serie)
    plt.plot(mavg)
    plt.title(title)
    plt.show()

def performance_metrics(results: pd.DataFrame) -> pd.DataFrame:
    # Compute the average reward for each algorithm
    res_per_algo = results.groupby(['algo', 'episode']).agg(
        {'reward': 'sum'}).reset_index()
    # Compute the diff in rewards
    pivoted = res_per_algo.pivot(
        index='episode', columns='algo', values='reward')
    pivoted['ddqn_vs_twap'] = pivoted['ddqn']-pivoted['twap']

    return pivoted


def create_data_and_env(path:str, sett:dict, test:float) -> tuple[pd.DataFrame, gym.Env]:
    """
    Function takes an input path with the data to generate a dataframe, splits
    it into training and test and creates a Gym environment for them. Returns
    the training and test datasets together with the Gym environments.
    """

    output_csv = f'{"/".join(path.split("/")[:-1])}/BTCUSD_PERP_agg_data.csv'
    df = load_btcusd(path, output_csv)

    # Split the data into training and testing
    testid = int(test*df.shape[0])
    
    train = df.iloc[:testid,:].reset_index(drop=True)
    test = df.iloc[testid:,:].reset_index(drop=True)
    
    train_env = TradeExecution(data=train, T=sett['t'], q0=sett['inventory'],
                    N=sett['steps'], alpha=sett['alpha'])
    
    test_env = TradeExecution(data=test, T=sett['t'], q0=sett['inventory'],
                    N=sett['steps'], alpha=sett['alpha'])
    
    return train, test, train_env, test_env

def action_masking(qty:int):
    return qty

def compute_perf(agent_rew:list, base_rew:list) -> np.array:
    """
    Computes PnL of agent rewards vs a baseline (mainly the twap)
    """
    if len(agent_rew)!=len(base_rew):
        raise ValueError(f'agent and baseline do not have same size, check!')
    
    gain = agent_rew/base_rew - 1.0
    return compute_stats(gain)

def compute_stats(pnls:np.array) -> dict:
    """
    Computes range of statistics over a numpy array
    """

    mean = np.mean(pnls)
    median = np.median(pnls)
    std = np.std(pnls)

    gain_loss_ratio  = np.mean(pnls[pnls>0])/np.mean(pnls[pnls<0])
    poss_ret = sum(pnls>0)/len(pnls)

    return {'mean':mean,'median':median, 'std':std,
            'glr':gain_loss_ratio, 'pos_ret':poss_ret}

def create_sett(nn_arch:list[int], act_in:bool, loss_func:str, opt_lr:float,
                   gamma:float, gr_step:float, greed_unif:bool,
                   env, episodes:int, buff_size:int, batch_size:int,
                   nn_cpy_cad:int, soft_update:float, logs:bool,
                   solv_metric, solv_tar:float, pretr:bool, 
                   max_poss_act, man_grad:bool) -> dict:
    
    def_nn_arch = {'neurons': nn_arch,
               'action_as_input':act_in,
                'loss_function': loss_func,
                'optimizer': tf.keras.optimizers.Adam(learning_rate=opt_lr),
                }

    def_agent = {'gamma': gamma,
                'greedy_step': gr_step,
                'greedy_uniform':greed_unif, 
                'environment': env,
                'episodes': episodes,
                'buff_size': buff_size, 
                'replay_mini_batch': batch_size,
                'nn_copy_cadency': nn_cpy_cad,
                'nn_architecture': def_nn_arch,
                'soft_update': soft_update,
                'add_logs':logs,
                'solve_metric':solv_metric,
                'solve_target':solv_tar,
                'pretrain':pretr,
                'max_poss_action':max_poss_act,
                'man_gradient':man_grad}

    return def_agent
    
def select_agent(label:str) -> DDQN:
    if label in ['ddqn', 'ddqn_pre']:
        ag=DDQN_tradexecution
    elif label=='twap':
        ag = TWAP
    elif label=='random':
        ag = RANDOM_TE
    
    return ag

def create_ids(k:int, size:int, ig_list:list) -> list[str]:
    """
    Creates a unique list of length=size with strings of size k
    from combinations of Ascii carachters and digits.
    """
    ids = set()
    while len(ids) != size:
        identifier = ''.join(random.choices(string.ascii_uppercase + string.digits, k=k))
        if identifier not in ig_list:
            ids.add(identifier)
    return list(ids)


def train_agent(id:str, sett:dict, agnt:str, env:gym.env,
                it:int, agent_path:str, logs_path:str) -> None:
    """
    Function formats settings and trains the agent.
    It does not return any variable, but stores
    the results of the training process in the
    dedicated folder
    """

    settings = create_sett(nn_arch=sett['neurons'], act_in=sett['act_as_in'],
                            loss_func=sett['loss_func'], opt_lr=sett['adam_lr'], 
                            gamma=sett['gamma'], gr_step=sett['grdy_step'],
                            greed_unif=sett['grdy_unif'], env=env,
                            episodes=sett['epi'], buff_size=sett['buff_size'],
                            batch_size=sett['batch_size'], nn_cpy_cad=sett['nn_cpy_cad'],
                            soft_update=sett['soft_update'], logs=sett['add_logs'],
                            solv_metric=sett['solv_metr'], solv_tar=sett['solv_tar'],
                            pretr=sett['pretrain'], max_poss_act=action_masking,
                            man_grad=sett['man_grad'])

    # Train the agents
    agent = select_agent(agnt)(sett=sett)
    agent.env.reset(0)
    if agnt.startswith('ddqn'):
        rew, loss, lgs, _, _ = agent.train()
    else:
        rew, loss, lgs = agent.train()

    # Compute stats
    if agnt != 'twap':
        # Retrieve the twap experiment reward curve with the same settings
        twap_sett_id =  create_trial_id(agent='twap', iter=it, params=settings)
        if is_exp_run(twap_sett_id, mode='train', path=agent_path):
            twap_rew  = load_nparray(filename=twap_sett_id, path=agent_path)
            perf_train = compute_perf(rew, twap_rew)
        else:
            raise ValueError('TWAP can not be retrieved since it has not been run yet'\
                                ' check code!')

    # Store rewards, losses, logs and performance as csv files
    write_nparray(perf_train, f'{id}_vs_twap_perf', logs_path)
    write_nparray(rew, f'{id}_rewards', logs_path)
    write_nparray(loss, f'{id}_losses', logs_path)
    write_pkl(lgs,f'{id}_logs', logs_path)

    # Save the agent and store the unique id 
    agent.save(agent_path, id)

# One of the issues that we face is that considering the chosen binomial distribution
# to greedily chose actions, we never explore the scenarios of selling all shares
# at the beginning of the interval for instance. It is for this reason that some
# researchers in the literature pre-train the model on some boundary cases, such as
# selling all shares at the beginning of the interval, or holding all shares till the
# end of the interval and then selling them all at once. The idea of this pre-training
# is to expose the algorithm to all possible conditions. Another approach will be to
# not use this binomial distribution when we are being greedy, but just draw actions
# from a uniform distribution