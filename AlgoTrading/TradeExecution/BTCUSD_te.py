from BTCUSD_te_utils import create_data_and_env, action_masking 
from BTCUSD_te_utils import compute_perf, create_sett, select_agent, create_ids
import os
import sys
sys.path.insert(1,'/home/axelbm23/Code/Library/')
import pandas as pd
from plotting import sns_lineplot

# Set up the default values
N_ITERATIONS = 3
GAMMA = 0.99
GREEDY_STEP = 999e-3
GREEDY_UNIF = True
EPISODES = 100 # 10_000 in DDQN paper
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

# Load BTC_USD perpetual data and create the environment
dirfile = os.path.abspath(os.path.dirname(__file__))
path = "/home/axelbm23/Code/AlgoTrading/data/BTCUSD_PERP*.csv"
env_sett = {'t': 30,  # time left to close our position, in seconds
            'inventory': 20,  # Initial inventory
            'steps': 5,  # number of steps to close this inventory
            # affects how much we penalize our algorithm to sell big chunks of shares.
            'alpha': 0.01
            }

train, test, env = create_data_and_env(path, env_sett, 0.7)

agents = ['twap', 'ddqn', 'ddqn_pre', 'random']
results = {}
stats = {}
logs = {}
saved_agent = {}
ids = create_ids(10, N_ITERATIONS*len(agents))
i = 0
for iter in range(N_ITERATIONS):
    for agent_i in agents:
        settings = create_sett(nn_arch=NEURONS, act_in=ACT_AS_IN, loss_func=LOSS_FUNC, 
                               opt_lr=ADAM_LR, gamma=GAMMA, gr_step=GREEDY_STEP,
                               greed_unif=GREEDY_UNIF, env=env, episodes=EPISODES,
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
        results[(agent_i, 'rew', iter)] = rew
        results[(agent_i, 'loss', iter)] = loss
        logs[(agent_i, iter)] = lgs

        # Compute stats
        if agent_i != 'twap':
            stats[(agent_i, iter)] = compute_perf(rew, results[('twap', 'rew', iter)])

        # Save the agent and store the unique id 
        agent.save(dirfile, ids[i])
        saved_agent[f'({agent_i}, {iter})'] = ids[i]

        i += 1

# Format output objects and write into a csv file
stats_df = pd.DataFrame.from_dict(stats, orient='index').reset_index()
stats_df.rename(columns={'level_0':'agent','level_1':'iteration'}, inplace=True)
stats_df.to_csv(f'{dirfile}/results/stats.csv')

# Convert dataframe into a tidy format
results_df = pd.DataFrame.from_dict(results).unstack().reset_index()
results_df.rename(columns={'level_0':'agent', 'level_1':'metric', 
                           'level_2':'iter', 'level_3':'episode',
                           0:'value'},
                  inplace=True)

# Compute average of metrics to smooth the graphs
#results_df['value(15_avg)'] = results_df.groupby(['agent', 'metric', 'iter'])['value'].rolling(15).mean().reset_index(drop=True)

# Plot the reward function of each of the agents
for iter in range(N_ITERATIONS):
    sns_lineplot(results_df.loc[((results_df['iter']==iter) & (results_df['metric']=='rew')),:],
                'episode','value', 'agent', f'agents reward vs episode (iter_{iter})')
    
# Goal_1: Check that indeed the ddqn converges to the more 
# or less the same profile than the TWAP


# Goal_2: ddqn_2 should behave worse than ddqn around the
# beginning of the training

# TO-DO: At the moment the episodes can be of different lengths. I have
# changed the logic in the environment so that all episodes have the same
# length.