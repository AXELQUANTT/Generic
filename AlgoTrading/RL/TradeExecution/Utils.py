from collections import deque
import datetime
import glob
from gymnasium import spaces
from gymnasium.core import RenderFrame
import gymnasium as gym
from itertools import groupby
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.layers import Dense
from keras.models import Sequential
from keras.activations import relu, linear, leaky_relu
from scipy.stats import binom
import time
from typing import Any
from typing import SupportsFloat


def plot(data: list, title: str, mavg: bool) -> None:
    serie = pd.Series(data)
    if mavg:
        mavg = serie.rolling(window=15).mean()
    plt.plot(serie)
    plt.plot(mavg)
    plt.title(title)
    plt.show()


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
        df.drop_duplicates(inplace=True)
        df.sort_values('datetime(s)', inplace=True)
        df.to_csv(output_csv)
    else:
        df = pd.read_csv(output_csv)
    return df


def format_history(hist: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(hist, columns=['qt', 't', 'action', 'reward', 'algo', 'episode'])


def twap_choose_action(curr_greedy, state, train):
    return int(te_env.q0/te_env.N)


def create_nn(input_size: int,
              model_params: dict,
              output_size: int,
              name: str) -> keras.Model:

    # Build the architecture
    tf.random.set_seed(1234)
    model = Sequential()
    model.name = name
    model.add(keras.Input(shape=(input_size,)))
    for neurons in model_params['neurons']:
        model.add(Dense(units=neurons, activation=relu,  # using leaky_relu instead of relu in order to avoid dying neuron issue in ReLU (acc to DDQN paper as well)
                  kernel_regularizer=tf.keras.regularizers.l2(model_params['lambda'])))

    # Finally add the output layer
    model.add(Dense(units=output_size, activation=linear))

    # Configure the network => specify loss functiona and optimizer
    model.compile(loss=model_params['loss_function'],
                  optimizer=model_params['optimizer'])
    # run_eagerly=True)

    return model


class TradingEnv(gym.Env):
    # FOR NOW:
    # Assumptions
    # 1. Given the size of the position in terms of the overall volume
    #   quoted at each level, we consider that the trader actions DO NOT
    #   directly effect the price process (S_t) during training. This means
    #   that regardless of the action taken, a, St does not change bc of that.
    # 2. For now we will consider only market orders in our approach,
    # 3. At any given point in time, our state is defined by the
    #   time index in which we are, t, and the remaining inventory q(t)
    #   to execute.
    #   t=0 at the beginning, t=T at the end.
    #   q(t=0) = q0 and q(t=T)= 0
    #   On a given time t, then the range of possible actions is restricted by
    #   (0, q(t)), which is discrete.
    # 4. We assume the agent makes decisions every T/N and that trades are executed
    #   on the subintervals of M_k.

    # For now we will only consider the current inventory and the time left for execution
    # as our state variables

    # The api we need to focus on the gym.Env class are the following:
    # def __init__()    => defines the initializations needed for the environment
    # def reset()       =>  this function resets the environment to its starting state
    #                       whenever the environment reaches a terminal state
    #                       or a state which is invalid. It needs to return the
    #                       state variable
    # def render()      =>  this function renders the data into a visual form, which
    #                       we will not use here.
    # def step()        =>  core part of the environemt. This is where we implement what
    #                       happens when the agent 'acts' on the environment. The
    #                       step function returns a tuple of four variables
    #                       (observation, reward, done, info).

    def __init__(self,
                 data: pd.DataFrame,
                 T: float,  # Time to close our position, in seconds
                 q0: int,  # Starting position
                 N: int,  # Number of steps over which the action can take a decision
                 alpha: float = 0.01,  # slippage factor
                 granularity: int = 0.25):

        self.data = data
        self.T = T
        self.q0 = q0
        self.N = N
        self.dt = self.T/self.N
        self.alpha = alpha
        self.mk = int(self.dt/granularity)
        self.data_idx = 0
        # For now we will only use to state variables, the inventory risk
        # and the current time.
        self.qt = self.q0
        self.t = self.T

        # This is the variable that controls whether we have reached a
        # terminal state, which can be triggered in one of the two
        # scenarios
        self.done = False

        # This is where we define what possible actions can the agent take
        # THIS NEEDS TO BE TAILORED TO THE ACTIONS WE ALLOW THE AGENT TO TAKE
        # MANDATORY TO DEFINE THIS ARGUMENT IN OUR gym class.
        # COMMENT:
        self.action_space = spaces.Discrete(n=self.qt+1, seed=42)

        # THIS NEEDS TO BE TAILORED TO WHAT WE DEFINE AS A STATE
        # MANDATORY TO DEFINE THIS IN OUR gym class.
        # In general, we define as a state everything that agent
        # can observe and that will use to take a given action.
        # For now our observation space will only be formed
        # by our current inventory, defined by self.qt
        # and the time left to execute, defined by self.t
        self.observation_space = spaces.Box(low=np.array([0, 0]),
                                            high=np.array([self.q0, self.T]),
                                            dtype=np.float64)

    def reset(self, id: int = 0) -> tuple[Any, dict[str, Any]]:
        self.qt = self.q0
        self.t = self.T
        self.done = False
        # in case the user provides it, the environment gets reset to this specific
        # timestamp
        if id > 0:
            if id > self.data.shape[0]-1:
                raise ValueError(f'Provided index is out of bounds, df.size={self.data.shape[0]},'
                                 f' index={id}')
            self.data_idx = id

        info = {'Inventory': self.qt,
                'Time_to_execute': self.t}

        return self.extract_state(), info

    # IMPORTANT => all columns that define a state should start with prefix state_
    def extract_state(self) -> np.array:
        return np.array(object=[self.qt, self.t], ndmin=2, dtype=int)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def _mask_action(self, action: Any) -> Any:
        if action > self.qt:
            return self.qt
        return action

    def _get_mid_price(self, idx: int) -> float:
        return self.data.loc[idx, 'mp']

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:

        # This is where we define of how the environment reacts
        # when agent takes a given action. So basically we have to code
        # the transition from s to s' when we take action a in state s

        # The first thing we will do is to asses whether the provided action is
        # a valid member of the action space defined by self.action_space
        assert self.action_space.contains(action)

        # How do we mask the action? Remember that at any given time t,
        # the agent can not sell more than the exisiting inventory qt.
        # Impose this condition
        self.action = self._mask_action(action)

        # We need to iterate over all sub-intervals comprised
        # between self.t and end_t
        reward = 0
        prev_qt = self.qt
        shs_per_interval = self.action/self.mk

        for _ in range(self.mk):
            curr_price = self._get_mid_price(self.data_idx)
            next_price = self._get_mid_price(self.data_idx+1)
            reward += self.qt*(next_price-curr_price) - \
                self.alpha*shs_per_interval**2
            self.data_idx += 1
            self.qt -= shs_per_interval

        # Since we are assuming that shares are sold evenly during the entire interval
        # it may be the case that due to rounding error self.qt != self.prev_qt - action.
        self.qt = int(prev_qt-self.action)

        # Update time left to execute
        self.t -= self.dt

        # Distribute some printing information
        info = {'Inventory': self.qt,
                'Time_to_execute': self.t}

        # If our inventory gets to 0 or we don't have more time to unfold our position,
        # we consider the episode to be finished
        if self.qt == 0:
            self.done = True

        elif self.t == 0:
            # If we get to the last step, add big penalty to close all remaining shares
            curr_price = self._get_mid_price(self.data_idx)
            next_price = self._get_mid_price(self.data_idx+1)
            reward += self.qt*(next_price-curr_price) - \
                self.alpha*(self.qt**2)
            self.data_idx += 1
            self.done = True

        # Note here observation needs to be the state s' which we have
        # arrived after applying action to state s
        obs = self.extract_state()

        # the result of step needs to be observation, reward, done, info
        return obs, reward, self.done, False, info


class ExpirienceReplay:
    def __init__(self, maxlen: int, mini_batch: int):
        self._buffer = deque(maxlen=maxlen)
        self.mini_batch = mini_batch

    # , p_init, p_last_step):
    def push(self, state, action, reward, next_state, terminated):
        self._buffer.append(
            (state, action, reward, next_state, terminated))

    def _get_batch(self):
        return random.sample(self._buffer, self.mini_batch)

    def get_arrays_from_batch(self):
        batch = self._get_batch()

        # Format arrays into 2d to make them easier to manipulate
        states = np.array([x[0] for x in batch])[:, 0, :]
        actions = np.array([x[1] for x in batch]).reshape(len(batch), 1)
        rewards = np.array([x[2] for x in batch]).reshape(len(batch), 1)
        next_states = np.array([x[3] for x in batch])[:, 0, :]
        terminated = np.array([x[4] for x in batch]).reshape(len(batch), 1)

        return states, actions, rewards, next_states, terminated

    def buffer_size(self):
        return len(self._buffer)


class DDQN():
    """
    This file devoted to implement a DQN (deep Q neuronal network)
    algorithm to be used as agent in any RL environment
    """

    def __init__(self,
                 sett: dict) -> None:

        self.gamma = sett['gamma']
        self.greedy = sett['greedy']
        self.env = sett['environment']
        self.episodes = sett['episodes']
        self.min_replay_size = sett['min_replay_size']
        self.replay_mini_batch = sett['replay_mini_batch']
        # The replay will contain four elements:
        # state => nx1 dimensional array containing all the quantities
        #         that constitute a state
        # a => any of the possible values allowed for our action
        # reward => float number with the reward for state s
        # state_prime => nx1 dimensional array containing the quantities
        #                defined for state s_prime (arrived state after applying
        #                action a to state s)
        self.replay_buffer = ExpirienceReplay(
            maxlen=self.min_replay_size, mini_batch=sett['replay_mini_batch'])
        # Create policy, used to extract actions, and target,
        # used to compute y_values, networks

        self.policy_nn: keras.Model = create_nn(input_size=self.env.extract_state().shape[1]+1,
                                                model_params=sett['nn_architecture'],
                                                output_size=sett['nn_architecture']['output_size'],
                                                name='policy')
        self.target_nn: keras.Model = create_nn(input_size=self.env.extract_state().shape[1]+1,
                                                model_params=sett['nn_architecture'],
                                                output_size=sett['nn_architecture']['output_size'],
                                                name='target')
        self.target_nn_initialized = True
        # Every how many episodes we copy the params from the policy
        # to the target network
        self.copy_cadency = sett['nn_copy_cadency']
        self.soft_update = sett['soft_update']

    def _soft_update_policy(self) -> None:
        """
        Function that specifies how the the params of the policy network need
        to be transfered to the target network.
        """

        old_weights = self.target_nn.get_weights()
        new_weights = self.policy_nn.get_weights()
        if self.target_nn_initialized:
            # If weights are still the initial ones, just set them
            # to the new weights
            target_weights = new_weights
            self.target_nn_initialized = False
        else:
            target_weights = list(self.soft_update*old_weights[idx]+(
                1.0-self.soft_update)*new_weights[idx] for idx in range(len(new_weights)))

        self.target_nn.set_weights(target_weights)

    def _compute_regressors_targets(self):
        """
        Given a set of transitions of state,action,reward,state',
        this function computes the the x and y ready to be consumed
        by the target and policy networks.
        """

        # First get all the data from the replay buffer
        states, actions, rewards, next_states, dones, = self.replay_buffer.get_arrays_from_batch()

        # Create the input for the NN adding the action to the state
        x = self._compute_input(states, actions)
        y = self._compute_targets(
            states, actions, rewards, next_states)

        return x, y

    def _agent_update(self) -> float:
        """
        Main function devoted to update the params
        of the policy and the target networks
        """

        if self.replay_buffer.buffer_size() < self.replay_mini_batch:
            # Ensure that we have enough data to start training
            return 0

        x, y = self._compute_regressors_targets()
        history = self.policy_nn.fit(x, y, verbose=0)
        return history.history['loss'][0]

    # The main function of the class
    def learn(self) -> list[float]:
        tot_ep_rewards = [0]*self.episodes
        tot_ep_losses = [0]*self.episodes
        n = self.episodes-1
        for ep in range(self.episodes):
            # Get the initial state
            # Why is it that we always get tranches of 60 movements? Isn't it the case that we never sell all remaining shares at once?
            print(f'Data_idx={self.env.data_idx}')
            self.env.reset()
            done = False
            ep_reward = 0
            greedy_param = ((n-ep)/n) * \
                (self.greedy[0]-self.greedy[1]) + self.greedy[1]
            loss = 0
            print(
                f'episode {ep}/{self.episodes-1}, greedy_param={round(greedy_param,3)}')
            while not done:
                s = self.env.extract_state()
                # Choose an action => is in this part that we have to apply the greedy policy
                # (in case we want to do it at all!). For sure this function needs to be dependant
                # on the state we are in!!!
                action = self.choose_action(
                    state=s, curr_greedy=greedy_param, train=True)
                # Get the action s_prime as result of taking action a in state s
                s_prime, reward, done, _, info = self.env.step(action)
                # Save all the needed quantites in our replay buffer. In order to do so, we'll
                # use Tensorflow replay buffer
                self.replay_buffer.push(
                    s, action, reward, s_prime, done)
                ep_reward += reward
                # After we have created a new data_point, update our networks in case
                # it's needed
                loss += self._agent_update()

            if ep % (self.copy_cadency-1) == 0.0:
                print('Copying weights from policy to target')
                # Finally assign the weights from our policy nn (now trained)
                # to the target nn
                self._soft_update_policy()

            # Update the max_reward acquired on this episode.
            # Compute the mean squared loss on the policy network
            tot_ep_rewards[ep] = ep_reward
            tot_ep_losses[ep] = loss

        return tot_ep_rewards, tot_ep_losses

    def _compute_input(self, states: np.array, actions: np.array = np.array([])) -> np.array:
        """
        This function creates the input needed for the nn, both for the
        target and for the policy. If no actions are provided, it generates
        all the possible actions (0, qt) for the current state.
        """

        if actions.size == 0:
            x = np.array([])
            for idx, state in enumerate(states):
                qt = state[0]
                mod_state = np.append(
                    np.array([idx]), states[idx:idx+1, :]).reshape(1, state.shape[0]+1)
                rep_states = np.repeat(mod_state, qt+1, axis=0)
                poss_actions = np.array(
                    range(qt+1), dtype=int).reshape(qt+1, 1)
                repeated_state = np.append(rep_states, poss_actions, axis=1)
                if x.size == 0:
                    x = repeated_state
                else:
                    x = np.concatenate([x, repeated_state])
            return x

        # Ensure output dimensions make sense
        n = states.shape[0]
        a = actions.shape[0]
        if n != a:
            raise ValueError(
                f'Dimensions of states={n} and actions={a} do not match, check!')

        return np.append(states, actions, axis=1)

    def _compute_targets(self, states, actions, rewards, next_states):
        """
        Auxiliary function devoted to compute the y-values for the networks
        """
        # max_q_s_prime_a_prime will be the max all predictions of policy network
        # over states s'. Note that in general, q_values are just the output values
        # for a given input state s over all actions that can be taken
        # Note that we are computing the q values from our target_nn, not our policy_nn
        x_next_states = self._compute_input(next_states)
        q_val = self.target_nn.predict(x_next_states[:, 1:], verbose=0)
        max_q_val = np.array([q_val[x_next_states[:, 0] == i].max()
                              for i, _ in groupby(x_next_states[:, 0])]).reshape(len(states), 1)

        y = rewards[:, 0] + + self.gamma * \
            (states[:, 1] > self.env.dt).astype(int)*max_q_val[:, 0]
        # r_last_step = next_states[:, 0] * (p_last_step[:, 0] -
        #                                   p_init[:, 0]) - self.env.alpha*next_states[:, 0]**2

        # y = rewards[:, 0] + self.gamma*(states[:, 1] > self.env.dt).astype(
        #    int)*max_q_val[:, 0] + self.gamma*(states[:, 1] == self.env.dt).astype(int)*r_last_step

        return y

    def choose_action(self, curr_greedy: float, state: np.array, train: bool) -> float:
        """
        Given an input state s, the agent will get the action it thinks
        is best suited for it. We have to differentiate between
        training and testing.

            -  In training we want to leave room for exploration => choose the suboptimal
            action for the seek of variability to train our agent. How much
            we allow the algo to explore depends on self.greedy parameter. Note that
            the degree to which we allow exploration decreases with the trained time.

            - In testing we want to perform at the best of our capabilities,
            so we only consider exploitation regime.
        """

        rand = np.random.uniform()
        # Remember state is a 2d np.array
        qt, t = state[0, :]
        if train and rand <= curr_greedy:
            # Exploration
            # Get a random trial from a binomial (qt , 1/time_to_expiry)
            # Why are we enforcing on the exploration phase a TWAP approach? Should
            # not the algorithm figure out that selling a lot of shares on an early
            # point is not the best. Seems to me this is a way to reduce the exploration
            # phase substantially => try with letting the algo pick any qt available
            return binom(qt, self.env.dt/t).rvs()

        # Exploitation
        # Compute the q-value from all the possible actions [0,qt] and retrieve the action
        # that delivers the best overall q-value
        x = self._compute_input(states=state)
        # Remember that the first element of x is just a mute index
        predictions = self.policy_nn.predict(x[:, 1:], verbose=0)
        best_action = np.argmax(predictions)

        return best_action

# LEAVE ALL THE FOLLOWING PARAMS UNTOUCHED
# replay_mini_batch => 32
# min_replay_size => 500
# episodes => 2_500
# soft_update => 0.99
# copy_cadency => 30

# 0.001 => sells all of them at the beginning of the interval
# 0.005 => sells all of them at the beginning of the interval
# 0.01 => too big, it sells all shares at the beginning of the interval
# 0.025 => too big, it sells all shares at the beginning of the interval
# 0.035 => too big, it sells all shares at the beginning of the interval

# So, it seems clear at this point that the algorithm is prefering to sell
# all shares at the beginning of the interval

# Changed logic so that we can start learning as soon as we have replay_mini_batch
# (instead of min_replay_size)

# Issues and limitations
# The first and most important one, is the huge dependency of the
# results in the latent parameter alpha. Small changes in the value of alpha
# translate to huge differences in the adopted policy by our agent.
# This can be linked to the fact that we are


# path = "/home/axelbm23/Code/AlgoTrading/RL/TradeExecution/data/1-09-1-20.csv"
path = "/home/axelbm23/Code/AlgoTrading/data/BTCUSD_PERP*.csv"
output_csv = f'{"/".join(path.split("/")[:-1])}/BTCUSD_PERP_agg_data.csv'
df = load_btcusd(path, output_csv)

settings = {'t': 60,  # time left to close our position, in seconds
            'inventory': 20,  # Initial inventory
            'steps': 5,  # number of steps to close this inventory
            # affects how much we penalize our algorithm to sell big chunks of shares.
            'alpha': 0.01
            }

# Create a trade execution environment with some random settings.
te_env = TradingEnv(data=df, T=settings['t'], q0=settings['inventory'],
                    N=settings['steps'], alpha=settings['alpha'])

network_architecture = {'neurons': [30]*5,  # same params as DDQN paper
                        'lambda': 0.01,  # set to the default value for now
                        'output_size': 1,
                        'loss_function': MeanSquaredError(),
                        # modified according to DDQN paper
                        'optimizer': Adam(learning_rate=0.0001),
                        }

# All settings for the agent are copied from DDQN paper
agent_settings = {'gamma': 0.99,
                  'greedy': [1.0, 0.01],
                  'environment': te_env,
                  'episodes': 2_000,  # 10_000 it's the param in DDQN
                  'min_replay_size': 5000,  # 5_000 it's the param in DDQN
                  'replay_mini_batch': 32,  # 32 is the value used in DDQN, and seems the most generic one
                  'nn_copy_cadency': 15,  # every how many episodes q_policy gets copied to q_target
                  'nn_architecture': network_architecture,
                  'soft_update': 0.99}


# Create our DDQN agent and train it
ddqn_agent = DDQN(sett=agent_settings)
rewards, losses = ddqn_agent.learn()

plot(data=rewards, title='rewards vs 15 period mavg', mavg=True)
plot(data=losses, title='losses vs 15 period mavg', mavg=True)

# When using only qt and t as state variables,
# one can show the result is purely deterministic

# Once the agent is trained, study how it reacts to different values
# inventory and time_to_expiry to see if it learned what we think it should
# 1. for the same level of inventory, it should sell more shares the closer
#   we are to the end of the interval.
# 2. for the same time to end of interval, it should sell more shares with
# increasing levels of inventory

history = []
print('Testing phase')
agent_chooser = {'twap': twap_choose_action,
                 'ddqn': ddqn_agent.choose_action}

# Just for testing purposes, we will implement an
# agent that follows a TWAP strategy.
episodes = 200
current_data_idx = te_env.data_idx
for agent in agent_chooser.keys():
    # Make sure that both agents are run over the exact same data
    te_env.reset(current_data_idx)
    for trie in range(episodes):
        te_env.reset()
        done = False
        while not done:
            state = te_env.extract_state()
            action = agent_chooser[agent](
                curr_greedy=0.01, state=state, train=False)
            obs, reward, done, _, info = te_env.step(action)
            history.append(list(state[0, :])+list([action]) +
                           list([reward, agent, trie]))

        print(f'progress...{trie+1}/{episodes}')

results = format_history(history)
