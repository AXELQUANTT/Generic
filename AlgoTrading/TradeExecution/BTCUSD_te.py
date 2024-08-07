import pandas as pd
import sys
from BTCUSD_te_utils import create_data_and_env, action_masking 
from BTCUSD_te_utils import compute_perf, create_sett, select_agent
sys.path.insert(1,'/home/axelbm23/Code/Library/')
from plotting import sns_lineplot

# Set up the default values
N_ITERATIONS = 3
GAMMA = 0.99
GREEDY_STEP = 999e-3
GREEDY_UNIF = True
EPISODES = 10 # 10_000 in DDQN paper
BUFF_SIZE = 1_000 # 5_000 in DDQN paper
BATCH_SIZE = 64 # 32 in DDQN paper
NN_COPY_CADENCY = None # 15 in DDQN paper
SOFT_UPDATE = 0.005
NEURONS = [20]*6 # 20 nodes, 6 layers in DDQN paper
ACT_AS_IN = True
ADD_LOGS = True
LOSS_FUNC = 'mean_squared_error'
ADAM_LR = 0.001
SOLV_METR = None
SOLV_TAR = -1
PRETRAIN = False

# Load BTC_USD perpetual data and create the environment
path = "/home/axelbm23/Code/AlgoTrading/data/BTCUSD_PERP*.csv"
env_sett = {'t': 30,  # time left to close our position, in seconds
            'inventory': 20,  # Initial inventory
            'steps': 5,  # number of steps to close this inventory
            # affects how much we penalize our algorithm to sell big chunks of shares.
            'alpha': 0.01
            }

data, env = create_data_and_env(path, env_sett)

results = {}
stats = {}
logs = {}
for iter in range(N_ITERATIONS):
    for agent_label in ['twap', 'ddqn', 'ddqn_pre', 'random']:
        settings = create_sett(nn_arch=NEURONS, act_in=ACT_AS_IN, loss_func=LOSS_FUNC, 
                               opt_lr=ADAM_LR, gamma=GAMMA, gr_step=GREEDY_STEP,
                               greed_unif=GREEDY_UNIF, env=env, episodes=EPISODES,
                               buff_size=BUFF_SIZE, batch_size=BATCH_SIZE,
                               nn_cpy_cad=NN_COPY_CADENCY, soft_update=SOFT_UPDATE,
                               logs=ADD_LOGS, solv_metric=SOLV_METR, solv_tar=SOLV_TAR,
                               pretr=PRETRAIN, max_poss_act=action_masking)
        
        settings['pretrain'] = agent_label=='ddqn_pre'
        agent = select_agent(agent_label)(sett=settings)
        agent.env.reset(0)
        if agent_label.startswith('ddqn'):
            rew, loss, lgs, _, _ = agent.train()
        else:
            rew, loss, lgs = agent.train()
        
        # Store output
        results[(agent_label, 'rew', iter)] = rew
        results[(agent_label, 'loss', iter)] = loss
        logs[(agent_label, iter)] = lgs

        # Compute stats
        if agent_label != 'twap':
            stats[(agent_label, iter)] = compute_perf(rew, results[('twap', 'rew', iter)])

# Format output objects
stats_df = pd.DataFrame.from_dict(stats, orient='index').reset_index()
stats_df.rename(columns={'level_0':'agent','level_1':'iteration'}, inplace=True)

# Set a multindex dataframe to store the results
results_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
results_df[['agent','metric', 'iteration']] = results_df['index'].tolist()
results_df.drop(['index'], inplace=True, axis=1)
results_df.set_index(['agent','metric','iteration'], inplace=True)
results_df = results_df.T