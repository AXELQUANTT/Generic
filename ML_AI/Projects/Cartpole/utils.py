import collections
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Any, Optional

def create_settings(key:str, val:Any, def_nn_arch:dict, def_agent:dict) -> Optional[dict]:
    # We are updating the optimizer each time we are creating a set of parameters because
    # we need to create different instances of it (i.e the same instance
    # of the same optimizer can not be shared among different networks)
    sett_nn = def_nn_arch.copy()
    sett_nn['optimizer'] = tf.keras.optimizers.Adam(learning_rate=0.001)
    match key:
        case 'nn_arch':
            sett_nn['neurons'] = val
            agent_sett = def_agent.copy()
        case 'lr':
            sett_nn['optimizer'] = tf.keras.optimizers.Adam(learning_rate=val)
            agent_sett = def_agent.copy()
        case 'target_update':
            agent_sett = def_agent.copy()
            agent_sett['nn_copy_cadency'] = val[0]
            agent_sett['soft_update'] = val[1]
        case 'greedy_step':
            agent_sett = def_agent.copy()
            agent_sett['greedy_step'] = val
        case 'default':
            agent_sett = def_agent.copy()
        case _:
            print(f'key is not valid, check')
    
    agent_sett['nn_architecture'] = sett_nn
    return agent_sett

def create_key(input:Any) -> str:
    if isinstance(input,collections.abc.Sequence):
        return "-".join([str(x) for x in input])
    
    return f'{input}'

def save_data(rewards,losses,exec_time,filename,out_path) -> None:
    exec_time_arr = [exec_time]*len(rewards)
    results_df = pd.DataFrame(data=[rewards,losses,exec_time_arr]).T
    results_df.columns = ['rewards','losses','exec_time']
    out_filename = f'{out_path}/{filename}.csv'
    results_df.to_csv(out_filename, index=False)

def solve_metric(rew) -> float:
    return np.average(rew[-100:])

def load_agent_results(out_path:str) -> dict[str,pd.DataFrame]:
    """
    Gets rewards and losses data from the output files and generates
    a list of DataFrames with all the information
    """
    
    files = glob.glob(f'{out_path}/*.csv')
    param_sett_ids = set(x.split('/')[-1].split('_iter')[0] for x in files)
    results = {}
    for param in param_sett_ids:
        # Select only the trials of this parameter settings
        files_for_this_param = [x for x in files if x.split('/')[-1].split('_iter')[0]==param]
        trials = []
        for trial_param in files_for_this_param:
            df = pd.read_csv(trial_param)
            df['iter'] = int(trial_param.split('_')[-1].split('.')[0])
            df.reset_index(inplace=True)
            df.rename(columns={'index':'episodes'},inplace=True)
            trials.append(df)
        df = pd.concat(trials)
        df.reset_index(drop=True,inplace=True)
        results[param] = df
    
    return results

def creage_avg(df, col, window) -> pd.DataFrame:
    """
    Computes moving averages of window size over the selected column col  
    """
    df[f'{col}({window})'] =  df[col].rolling(window).mean()
    return df