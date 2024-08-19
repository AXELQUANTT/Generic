from BTCUSD_te_utils import create_data_and_env, action_masking 
from BTCUSD_te_utils import compute_perf, create_sett, select_agent, create_ids
import os
import sys
sys.path.insert(1,'/home/axelbm23/Code/Library/')
import numpy as np
import pandas as pd
from plotting import sns_lineplot

# Set up the default values
N_ITERATIONS = 1
GAMMA = 0.99
GREEDY_STEP = 999e-3
GREEDY_UNIF = True
EPISODES = 50 # 10_000 in DDQN paper
TEST_EPISODES = int(EPISODES*0.5)
BUFF_SIZE = 1_000 # 5_000 in DDQN paper
BATCH_SIZE = 10 # 32 in DDQN paper
NN_COPY_CADENCY = None # 15 in DDQN paper
SOFT_UPDATE = 0.005
NEURONS = [20]*6 # 20 nodes, 6 layers in DDQN paper
ACT_AS_IN = True
ADD_LOGS = True
LOSS_FUNC = 'mean_squared_error'
ADAM_LR = 0.001
SOLV_METR = None
SOLV_TAR = -1
PRETRAIN = 10
MAN_GRAD = True
EPISODES_TEST = 500
SAVEMODEL = True

# Load BTC_USD perpetual data and create the environment
dirfile = os.path.abspath(os.path.dirname(__file__))
sav_ag_path = f'{dirfile}/results/agents/'
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

train, test, train_env, test_env = create_data_and_env(path, env_sett, 0.7)

agents = ['twap', 'ddqn', 'ddqn_pre', 'random']
results_train = {}
stats_train = {}
logs_train = {}
i = 0

if SAVEMODEL:
    saved_agent = {}
    ids = create_ids(10, N_ITERATIONS*len(agents))

for iter in range(N_ITERATIONS):
    for agent_i in agents:
        settings = create_sett(nn_arch=NEURONS, act_in=ACT_AS_IN, loss_func=LOSS_FUNC, 
                               opt_lr=ADAM_LR, gamma=GAMMA, gr_step=GREEDY_STEP,
                               greed_unif=GREEDY_UNIF, env=train_env, episodes=EPISODES,
                               buff_size=BUFF_SIZE, batch_size=BATCH_SIZE,
                               nn_cpy_cad=NN_COPY_CADENCY, soft_update=SOFT_UPDATE,
                               logs=ADD_LOGS, solv_metric=SOLV_METR, solv_tar=SOLV_TAR,
                               pretr=PRETRAIN, max_poss_act=action_masking, man_grad=MAN_GRAD)
        
        # Train the agents
        settings['pretrain'] = agent_i == 'ddqn_pre'
        agent = select_agent(agent_i)(sett=settings)
        agent.env.reset(0)
        if agent_i.startswith('ddqn'):
            rew, loss, lgs, _, _ = agent.train()
        else:
            rew, loss, lgs = agent.train()

        # Store output
        results_train[(agent_i, 'rew', iter)] = rew
        results_train[(agent_i, 'loss', iter)] = loss
        logs_train[(agent_i, iter)] = lgs

        # Compute stats
        if agent_i != 'twap':
            stats_train[(agent_i, iter)] = compute_perf(rew, results_train[('twap', 'rew', iter)])

        if SAVEMODEL:
            # Save the agent and store the unique id 
            agent.save(sav_ag_path, ids[i])
            saved_agent[f'({agent_i}, {iter})'] = ids[i]

        i += 1

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

# Compute average of metrics to smooth the graphs
#results_df['value(15_avg)'] = results_df.groupby(['agent', 'metric', 'iter'])['value'].rolling(15).mean().reset_index(drop=True)

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
            test_env.reset()

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
logs_df = pd.DataFrame.from_dict(logs_test).unstack().reset_index()
logs_df.rename(columns={'level_0':'agent', 'level_1':'iter'}, inplace=True)
logs_df[['state','action','reward', 'state_prime', 'timestamp']] = pd.DataFrame(logs_df[0].tolist())
logs_df.drop(['level_2', 0], axis=1, inplace=True)
logs_df.to_csv(f'{dirfile}/results/test_logs.csv')