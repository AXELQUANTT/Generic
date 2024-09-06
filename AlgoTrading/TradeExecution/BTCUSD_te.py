from BTCUSD_te_utils import create_data_and_env, action_masking , create_trial_id
from BTCUSD_te_utils import compute_perf, create_sett, select_agent, create_ids
from BTCUSD_te_utils import update_sett, is_exp_run, get_ids, write_nparray, load_nparray
from BTCUSD_te_utils import write_pkl, load_pkl, train_agent
import os
import sys
sys.path.insert(1,'/home/axelbm23/Code/Library/')
import numpy as np
import pandas as pd
from plotting import sns_lineplot

# Set up the common settings
common_sett = {'n_iter':3,
                'gamma':0.99,
                'grdy_step':0.998,
                'grdy_unif':False,
                'epi':4_500, #10_000 in DDQN paper
                'test_epi':450,
                'buff_size':1_000, #5_000 in DDQN paper
                'batch_size':32, #32 in DDQN paper
                'nn_cpy_cad':None,
                'soft_update':0.005,
                'neurons':[20]*6, #20 nodesx 6 layers in DDQN paper
                'act_as_in':True,
                'add_logs':True,
                'loss_func':'mean_squared_error',
                'adam_lr':0.001,
                'solv_metr':None,
                'solv_tar':-1,
                'pretrain':1_000,
                'man_grad':True}

# Load BTC_USD perpetual data and create the environment
dirfile = os.path.abspath(os.path.dirname(__file__))
ag_path = f'{dirfile}/results/agents'
logs_path = f'{dirfile}/results/logs'
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
#results_train = {}
#stats_train = {}
#logs_train = {}
i = 0

# What parameters will we play with?
# nn_architecture
# greedy_step
# adam_lr
# copy type
study_sett = {'neurons':[[20]*3, [20]*4],
              'grdy_step':[0.999, 0.995],
              'adam_lr':[0.0001, 0.0005],
              'target_copy':[[None, 0.01], [None, 0.05], [15, None], [150, None]],
              'default':None}

# The idea is to get an unique alphanumeric value for 
# the set of parameters we have chosen, which is more
# convenient than a very long string
for agent_i in agents:
    if agent_i.startswith('ddqn'):
        # Update params with the study sett
        for par_key,par_val in study_sett.items():
            exp_sett = update_sett(common_sett, par_key, par_val)
            for it in range(common_sett['n_iter']):
                sett_id = create_trial_id(agent=agent_i, params=exp_sett, iter=it)
                id = is_exp_run(id=sett_id, mode='train', path=ag_path)
                if not id:
                    # Create a unique identifier, making sure
                    # it was not used by another experiment
                    id = create_ids(k=10, size=1, ignoring_list=get_ids(path=ag_path).values())
                    train_agent(id=id, sett=exp_sett, agnt=agent_i, env=train_env, it=it, agent_path=ag_path, logs_path=logs_path)
                    
    elif agent_i=='twap':
        # Deterministic agent, so only need to run it once
        sett_id = create_trial_id(agent=agent_i, params=exp_sett)
        id = is_exp_run(id=sett_id, mode='train', path=ag_path)
        if not id:
            id = create_ids(k=10, size=1, ignoring_list=get_ids(path=ag_path).values())
            train_agent(id=id, sett=exp_sett, agnt=agent_i, env=train_env, it=it, agent_path=ag_path, logs_path=logs_path)

    elif agent_i=='random':
        # Since it's a random agent, the outcome will change
        # trial to trial
        for it in range(common_sett['n_iter']):
            sett_id = create_trial_id(agent=agent_i, params=exp_sett, iter=it)
            id = is_exp_run(id=sett_id, mode='train', path=ag_path)
            if not id:
                id = create_ids(k=10, size=1, ignoring_list=get_ids(path=ag_path).values())
                train_agent(id=id, sett=exp_sett, agnt=agent_i, env=train_env, it=it, agent_path=ag_path, logs_path=logs_path)

# TO-DO: Retrieve all the results from this parameter settings choice and 
# create the output csv files

# Format output objects and write into a csv file
train_stats_df = pd.DataFrame.from_dict(stats_train, orient='index').reset_index()
train_stats_df.rename(columns={'level_0':'agent','level_1':'iteration'}, inplace=True)
train_stats_df.to_csv(f'{dirfile}/results/train_stats.csv')

# Convert dataframe into a tidy format
results_train_df = pd.DataFrame.from_dict(results_train).unstack().reset_index()
results_train_df.rename(columns={'level_0':'agent', 'level_1':'metric', 
                           'level_2':'iter', 'level_3':'episode',
                           0:'value'},
                  inplace=True)

logs_train_df = pd.DataFrame.from_dict(logs_train).unstack().reset_index()
logs_train_df.rename(columns={'level_0':'agent', 'level_1':'iter'}, inplace=True)
logs_train_df[['state','action','reward', 'state_prime', 'timestamp']] = pd.DataFrame(logs_train_df[0].tolist())
logs_train_df.drop(['level_2', 0], axis=1, inplace=True)
logs_train_df.to_csv(f'{dirfile}/results/train_logs.csv')


# Plot the reward function of each of the agents
for iter in range(N_ITERATIONS):
    sns_lineplot(results_train_df.loc[((results_train_df['iter']==iter) & (results_train_df['metric']=='rew')),:],
                'episode','value', 'agent', f'agents reward vs episode (iter_{iter})')

logs_test = {}
stats_test = {}
results_test = {}
# Load the models and apply the learnt policy to the test data
for iter in range(N_ITERATIONS):
    for agent_i in agents:
        # Initialize and load the model
        model = select_agent(agent_i)(sett=settings)
        model = model.load(sav_ag_path, saved_agent[f'({agent_i}, {iter})'])
        rew = []
        # Make sure env is always set to the start for each  agent iteration
        s,_ = test_env.reset(0)
        data = []
        for ep in range(TEST_EPISODES):
            ep_rew = 0
            done = False
            while not done:
                action = model.choose_action(state=s, train=False)# Issue seems that done condition is not satisfied
                s_prime, reward, done, _, info = test_env.step(action)
                data.append([s, action, reward, s_prime, info['timestamp']])
                s = s_prime
                ep_rew += reward

            rew.append(ep_rew)
            # Reset environment as episode has concluded, so inventory and
            # rest of quantities need to be reset. Do not reset at index 0
            # though, as we want to run on different data
            s,_ = test_env.reset()

        logs_test[(agent_i, iter)] = data
        rew = np.array(rew)
        results_test[(agent_i, 'rew', iter)] = rew
        if agent_i != 'twap':
            stats_test[(agent_i, iter)] = compute_perf(rew, results_test[('twap', 'rew', iter)])

# Compute test statistics
test_stats_df = pd.DataFrame.from_dict(stats_test, orient='index').reset_index()
test_stats_df.rename(columns={'level_0':'agent','level_1':'iteration'}, inplace=True)
test_stats_df.to_csv(f'{dirfile}/results/test_stats.csv')

# Convert dataframe into a tidy format
results_test_df = pd.DataFrame.from_dict(results_test).unstack().reset_index()
results_test_df.rename(columns={'level_0':'agent', 'level_1':'metric', 
                           'level_2':'iter', 'level_3':'episode',
                           0:'value'},
                  inplace=True)

for iter in range(N_ITERATIONS):
    sns_lineplot(results_test_df.loc[((results_test_df['iter']==iter) & (results_test_df['metric']=='rew')),:],
                'episode','value', 'agent', f'Test: agents reward vs episode (iter_{iter})')

# Compute statistics and graphs and save test results
logs_test_df = pd.DataFrame.from_dict(logs_test).unstack().reset_index()
logs_test_df.rename(columns={'level_0':'agent', 'level_1':'iter'}, inplace=True)
logs_test_df[['state','action','reward', 'state_prime', 'timestamp']] = pd.DataFrame(logs_test_df[0].tolist())
logs_test_df.drop(['level_2', 0], axis=1, inplace=True)
logs_test_df.to_csv(f'{dirfile}/results/test_logs.csv')


# At this point there's quite a huge variance on the results of the experiment, which means
# the learning is not as stable as we would like. One of the things to note is that
# almost none of the experiments leads to a learning that is able to close our entire
# position, so maybe try changing the alpha parameter. This were the results
# obtained with choosing actiosn following a uniform distribution. 
# Choose them according to our binomial, it should stabilize training.