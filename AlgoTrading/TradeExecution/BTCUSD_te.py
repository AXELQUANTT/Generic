from BTCUSD_te_utils import create_data_and_env, action_masking
from BTCUSD_te_utils import compute_perf, create_sett, select_agent
from BTCUSD_te_utils import write_nparray, load_nparray
from BTCUSD_te_utils import write_pkl, load_pkl,format_stats
from BTCUSD_te_utils import format_results, format_logs, prep_sett_and_launch_learn
import os
import sys
sys.path.insert(1,'/home/axelbm23/Code/Library/')
import numpy as np
import pandas as pd
from plotting import sns_lineplot

# Set up the common settings
common_sett = {'n_iter':3,
                'gamma':0.99,
                'greedy_step':0.998,
                'greedy_uniform':False,
                'episodes':10_00,#4_500, #10_000 in DDQN paper
                'test_epi':2_000,
                'buff_size':1_000,#5_000 in DDQN paper
                'replay_mini_batch':32, #32 in DDQN paper
                'nn_copy_cadency':None,
                'soft_update':0.005,
                'neurons':[20]*6, #20 nodes x 6 layers in DDQN paper
                'act_as_in':True,
                'add_logs':True,
                'loss_func':'mean_squared_error',
                'adam_lr':0.001,
                'solve_metric':None,
                'solve_target':-1,
                'pretrain':1_000,
                'man_gradient':True}

# Load BTC_USD perpetual data and create the environment
dirfile = os.path.abspath(os.path.dirname(__file__))
gen_path = f'{dirfile}/results' 
ag_path = f'{gen_path}/agents'
logs_path = f'{gen_path}/logs'
path = "/home/axelbm23/Code/AlgoTrading/data/BTCUSD_PERP*.csv"
env_sett = {'t': 30,  # time left to close our position, in seconds
            'inventory': 20,  # Initial inventory
            'steps': 5,  # number of steps to close this inventory
            # affects how much we penalize our algorithm to sell big chunks of shares.
            'alpha': 0.01
            }

# TO-DO: Analyzing the timestamps, there's 30s between updates. So every 30s
# we take an action. This is not what I had in mind, I thought 30s was the overall
# time to close our entire position.

train, test, train_env, test_env = create_data_and_env(path, env_sett, 0.8)
agents = ['twap', 'ddqn', 'ddqn_pre', 'random']

# What parameters will we play with?
# nn_architecture
# greedy_step
# adam_lr
# copy type
study_sett = {'neurons':[[20]*3, [20]*4],
              'greedy_step':[0.999, 0.995],
              'adam_lr':[0.0001, 0.0005],
              'target_copy':[[None,0.01], [None, 0.05], [15, None], [150, None]],
              'default':None}

# The idea is to get an unique alphanumeric value for 
# the set of parameters we have chosen, which is more
# convenient than a very long string
results_train = {}
logs_train = {}
stats_train = {}
experiments = {}
for agent_i in agents:
    if agent_i=='ddqn':
        # All the experiments should only be with ddqn, not ddqn_pre. Change this logic
        # Update params with the study sett
        for par_key, par_val in study_sett.items():
            # par_val can be an array of elements, iterate over it
            if par_val:
                for ind_sett in par_val:
                   results_train, logs_train, stats_train, experiments = prep_sett_and_launch_learn(iterations=common_sett['n_iter'], 
                                                                                       def_sett=common_sett,new_key=par_key,
                                                                                       new_val=ind_sett, agnt=agent_i, path=gen_path,
                                                                                       env=train_env, results=results_train,
                                                                                       logs=logs_train, stats=stats_train, exp=experiments)
                   
            else:
                results_train, logs_train, stats_train, experiments = prep_sett_and_launch_learn(iterations=common_sett['n_iter'], def_sett=common_sett,
                                               new_key=par_key, new_val=par_val, agnt=agent_i,
                                               path=gen_path, env=train_env,
                                               results=results_train, logs=logs_train, stats=stats_train,
                                               exp=experiments)
                    
    elif agent_i=='twap':
        results_train, logs_train, stats_train, experiments = prep_sett_and_launch_learn(iterations=common_sett['n_iter'], def_sett=common_sett,
                                               new_key='default', new_val=None, agnt=agent_i,
                                               path=gen_path, env=train_env, results=results_train,
                                               logs=logs_train, stats=stats_train, exp=experiments)

    elif agent_i in ['random', 'ddqn_pre']:
        results_train, logs_train, stats_train, experiments = prep_sett_and_launch_learn(iterations=common_sett['n_iter'], def_sett=common_sett,
                                               new_key='default', new_val=None, agnt=agent_i,
                                               path=gen_path, env=train_env, results=results_train,
                                               logs=logs_train, stats=stats_train, exp=experiments)
    else:
        raise ValueError(f'Unknown agent, {agent_i}, please check implementation')

# TO-DO: Issue, we need to add the id of the experiment, as different
# experiments are run for the same agent
stats_train_df = format_stats(stats_train, 'train_stats', gen_path)
results_train_df = format_results(results_train, 'train_results', gen_path)
logs_train_df = format_logs(logs_train, 'train_logs', gen_path)

# Plot the reward function of each of the agents
for iter in range(common_sett['n_iter']):
    sns_lineplot(results_train_df.loc[((results_train_df['iter']==iter) & (results_train_df['metric']=='rew')),:],
                'episode','value', 'agent', f'agents reward vs episode (iter_{iter})')

logs_test = {}
stats_test = {}
results_test = {}
# Load the models and apply the learnt policy to the test data
test_sett = create_sett(nn_arch=common_sett['neurons'], act_in=common_sett['act_as_in'],
                            loss_func=common_sett['loss_func'], opt_lr=common_sett['adam_lr'], 
                            gamma=common_sett['gamma'], gr_step=common_sett['greedy_step'],
                            greed_unif=common_sett['greedy_uniform'], env=test_env,
                            episodes=common_sett['test_epi'], buff_size=common_sett['buff_size'],
                            batch_size=common_sett['replay_mini_batch'], nn_cpy_cad=common_sett['nn_copy_cadency'],
                            soft_update=common_sett['soft_update'], logs=common_sett['add_logs'],
                            solv_metric=common_sett['solve_metric'], solv_tar=common_sett['solve_target'],
                            pretr=common_sett['pretrain'], max_poss_act=action_masking,
                            man_grad=common_sett['man_gradient'])
n = len(experiments)
i = 1
for agent_i in agents:
    sel_exps = dict((sett_id, id) for sett_id,id in experiments.items() if sett_id.split('-')[0]==agent_i)
    for sett_id, id in sel_exps.items():
        if not os.path.exists(f'{logs_path}/{id}_rewards_test.npy'):
            model = select_agent(agent_i)(sett=test_sett).load(ag_path, id)
            rew, logs = model.test(env=test_env, episodes=common_sett['test_epi'])
            # Write output
            write_nparray(rew, f'{id}_rewards_test', logs_path)
            write_pkl(logs, f'{id}_logs_test', logs_path)
        else:
            print('Loading test results')
            rew = load_nparray(f'{id}_rewards_test', logs_path)
            logs = load_pkl(f'{id}_logs_test', logs_path)

        # Save the output. 
        # In case it's twap, then it's the same for all three iterations
        if agent_i == 'twap':
            for iter in range(common_sett['n_iter']):
                results_test[(agent_i, 'rew', iter, sett_id)] = rew
                logs_test[(agent_i, iter, sett_id)] = logs
        else:
            twap_key = [key for key in results_test.keys() if key[0]=='twap' and key[1]=='rew' and key[2]==0][0]
            iter = int(sett_id.split('-')[1].split('_')[1])
            results_test[(agent_i, 'rew', iter, sett_id)] = rew
            logs_test[(agent_i, iter, sett_id)] = logs
            stats_test[(agent_i, iter, sett_id)] = compute_perf(rew, results_test[twap_key])
        
        print(f'Progress on the test data ... {round(i/n,3)}')
        i += 1

# Format and save test statistics
test_stats_df = format_stats(stats_test, 'test_stats', gen_path)
test_results_df = format_results(results_test,'test_results', gen_path)
test_logs_df = format_logs(logs_test, 'test_logs', gen_path)

for iter in range(common_sett['n_iter']):
    sns_lineplot(test_results_df.loc[((test_results_df['iter']==iter) & (test_results_df['metric']=='rew')),:],
                'episode','value', 'agent', f'Test: agents reward vs episode (iter_{iter})')