#import glob
#import gymnasium as gym
import matplotlib.pyplot as plt
#import numpy as np
import os
#import pandas as pd
#import seaborn as sns
import sys
sys.path.insert(1,'/home/axelbm23/Code/ML_AI/Algos/ReinforcementLearning/')
from agents import DDQN_tradexecution, TWAP, RANDOM_TE
import time
import tensorflow as tf
#import tf_keras
from typing import Any,Optional
from BTCUSD_te_utils import create_data_and_env, action_masking, format_logs, compute_perf

# Set up the default values
N_ITERATIONS = 3
GAMMA = 0.99
GREEDY_STEP = 999e-3
EPISODES = 50 # 10_000 in DDQN paper
BUFF_SIZE = 1_000 # 5_000 in DDQN paper
BATCH_SIZE = 64 # 32 in DDQN paper
NN_COPY_CADENCY = None # 15 in DDQN paper
SOFT_UPDATE = 0.005
NEURONS = [128]*2
ACT_AS_IN = True
ADD_LOGS = True
LOSS_FUNC = 'mean_squared_error'
ADAM_LR = 0.001
OUTPUT_PATH = f'{os.getcwd()}/results/rewards_losses'

# Load BTC_USD perpetual data and create the environment
path = "/home/axelbm23/Code/AlgoTrading/data/BTCUSD_PERP*.csv"
env_sett = {'t': 30,  # time left to close our position, in seconds
            'inventory': 20,  # Initial inventory
            'steps': 5,  # number of steps to close this inventory
            # affects how much we penalize our algorithm to sell big chunks of shares.
            'alpha': 0.01
            }

data, env = create_data_and_env(path, env_sett)

def_nn_arch = {'neurons': NEURONS,
               'action_as_input':ACT_AS_IN,
                'loss_function': LOSS_FUNC,
                'optimizer': tf.keras.optimizers.Adam(learning_rate=ADAM_LR),
                }

def_agent = {'gamma': GAMMA,
            'greedy_step': GREEDY_STEP,
            'greedy_uniform':True,
            'environment': env,
            'episodes': EPISODES,
            'buff_size': BUFF_SIZE, 
            'replay_mini_batch': BATCH_SIZE,
            'nn_copy_cadency': NN_COPY_CADENCY,
            'nn_architecture': def_nn_arch,
            'soft_update': SOFT_UPDATE,
            'add_logs':ADD_LOGS,
            'solve_metric':None,
            'solve_target':-1,
            'pretrain':False,
            'max_poss_action':action_masking}

# Train our DDQN agent
ddqn_agent_no_pretrain = DDQN_tradexecution(sett=def_agent)
ddqn_rewards, ddqn_losses, ddqn_logs, _, _ = ddqn_agent_no_pretrain.train()
#ddqn_df = format_logs(ddqn_logs)

# Train TWAP agent
twap_agent = TWAP(sett=def_agent)
# Make sure that environmnet is at the beginning again
twap_agent.env.reset(0)
twap_rewards, twap_losses, twap_logs = twap_agent.train()

# RANDOM agent
random_agent = RANDOM_TE(sett=def_agent)
# Make sure that environmnet is at the beginning again
random_agent.env.reset(0)
random_rewards, random_losses, random_logs = random_agent.train()

# DDQN agent, pre-training it this time
def_agent['pretrain'] = True
ddqn_pre_trained = DDQN_tradexecution(sett=def_agent)
random_agent.env.reset(0)
ddqn_pre_rewards, ddqn_pre_losses, ddqn_pre_logs, start, end = ddqn_agent_no_pretrain.train()

# Compute performance metrics over the multiple agents trained
ddqn_nopre_perf = compute_perf(ddqn_rewards, twap_rewards)
ddqn_pre_perf = compute_perf(ddqn_pre_rewards, twap_rewards)
rand_perf = compute_perf(random_rewards, twap_rewards)

# OBS_1: How come ddqn_pre always sells all shares at the beginning
# of the interval? This is caused by the binomial distribution we are
# using for the e-greedy approach. Use a uniform distribution instead