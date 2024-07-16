from collections import deque
from gymnasium import spaces
from gymnasium.core import RenderFrame
import gymnasium as gym
from itertools import groupby
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sys
from scipy.stats import binom
import tensorflow as tf
import time
import tf_agents
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
import tf_keras.activations
from tf_keras.activations import relu,linear
from typing import Any
from typing import SupportsFloat

def create_dense_layer(neurons:int,
                 activation:tf_keras.activations=relu,
                 l2_reg=None) -> tf_keras.layers.Dense:
    
    return tf.keras.layers.Dense(units=neurons, activation=activation, 
                  kernel_regularizer=l2_reg)

def create_nn(input_size: int,
              model_params: dict,
              output_size: int) -> tf_keras.Model:

    # Build the architecture
    #tf.random.set_seed(1234)
    model = tf.keras.Sequential()
    #if name!='':
    #    model.name = name
    if input_size!=0:
        model.add(tf.keras.Input(shape=(input_size,)))
    for neurons in model_params['neurons']:
        model.add(create_dense_layer(neurons=neurons,
                                     activation=relu))

    # Finally add the output layer
    model.add(create_dense_layer(neurons=output_size, activation=linear))

    # Configure the network => specify loss functiona and optimizer
    #if compile:
    model.compile(loss=model_params['loss_function'],
                    optimizer=model_params['optimizer'])

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

    def _get_mp_diff(self, idx: int) -> float:
        return self.data.loc[idx, 'mp_diff']

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
            reward += self.qt * \
                self._get_mp_diff(self.data_idx) - \
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
            # If we get to the last step, add big penalty for all the remaining
            # shares that we have
            reward += self.qt * \
                self._get_mp_diff(self.data_idx) - \
                self.alpha*shs_per_interval**2
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
        # Make sure that state and next_state are 2d arrays
        if state.ndim==1:
            state = state.reshape(1,len(state))
            next_state = next_state.reshape(1,len(next_state))
        
        self._buffer.append(
            (state, action, reward, next_state, terminated))

    def _get_batch(self):
        return random.sample(self._buffer, self.mini_batch)

    def get_arrays_from_batch(self):
        batch = self._get_batch()

        # Check the dimensionality of the states
        states = np.array([x[0] for x in batch])[:, 0, :]
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])[:, 0, :]
        terminated = np.array([x[4] for x in batch])

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
        # TO-DO: Maybe we should stop being greedy after a number of episodes. Otherwise
        # we can never train the model with what it is learning as we always pick a random
        # action => instead of the one that maximizes reward
        self.greedy = sett['greedy']
        self.unif = sett['greedy_uniform']
        self.greedy_max_step = sett['greedy_max_step']
        self.env = sett['environment']
        self.episodes = sett['episodes']
        self.buff_size = sett['buff_size']
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
            maxlen=self.buff_size, mini_batch=self.replay_mini_batch)
        # Create policy, used to extract actions, and target,
        # used to compute y_values, networks
        self.action_as_in = sett['action_as_input']

        self.policy_nn: tf_keras.Model = create_nn(input_size=self.env.observation_space.shape[0]+1 if self.action_as_in else self.env.observation_space.shape[0],
                                                model_params=sett['nn_architecture'],
                                                output_size=sett['nn_architecture']['output_size'])
        
        self.target_nn: tf_keras.Model = create_nn(input_size=self.env.observation_space.shape[0]+1 if self.action_as_in else self.env.observation_space.shape[0],
                                                model_params=sett['nn_architecture'],
                                                output_size=sett['nn_architecture']['output_size'])
        
        self.policy_nn.compile(loss=sett['nn_architecture']['loss_function'],
                    optimizer=sett['nn_architecture']['optimizer'])
        self.target_nn.compile(loss=sett['nn_architecture']['loss_function'],
                    optimizer=sett['nn_architecture']['optimizer'])
        
        self.target_nn_initialized = True
        self.already_copied = False
        # Every how many episodes we copy the params from the policy
        # to the target network
        self.copy_cadency = sett['nn_copy_cadency']
        self.soft_update = sett['soft_update']
        self.pretrain = sett['pretrain']
        self.custom_exploration = sett['cust_exploration']
        self.add_log = sett['add_logs']

    def _soft_update_policy(self, ep:int) -> None:
        """
        Function that specifies how the the params of the policy network need
        to be transfered to the target network.
        """

        old_weights = self.target_nn.get_weights()
        new_weights = self.policy_nn.get_weights()
        if self.target_nn_initialized:
            # If weights are still the initial ones, just set them
            # to the new weights
            print(f'Hard copy policy_weights to target_weights')
            self.target_nn.set_weights(new_weights)
            self.target_nn_initialized = False
            return 
        
        if self.copy_cadency:
            # Check if episode is the one in which we need to copy it
            if (ep+1) % self.copy_cadency == 0.0:
                if not self.already_copied:
                    print(f'Hard copy policy_weights to target_weights')
                    self.target_nn.set_weights(new_weights)
                    self.already_copied=True
                return
            
            self.already_copied = False
            return
        
        print('Copying weights using soft_update')
        # if self.copy_cadency is None, it means we need to update via Tau (soft_update)
        target_weights = list((1-self.soft_update)*old_weights[idx]+\
                            self.soft_update*new_weights[idx] for idx in range(len(new_weights)))
        self.target_nn.set_weights(target_weights)
        

    def _compute_regressors_targets(self):
        """
        Given a set of transitions of state,action,reward,state',
        this function computes the the x and y ready to be consumed
        by the target and policy networks.
        """
        states, actions, rewards, next_states, dones = self.replay_buffer.get_arrays_from_batch()

        # Create the input for the NN adding the action to the state
        x = states
        if self.action_as_in:
            x = self._compute_input(states, actions)
        y = self._compute_targets(
            states, actions, rewards, next_states, dones)

        return x, y

    def _agent_update(self) -> float:
        """
        Main function devoted to update params of the policy network
        """
        if self.replay_buffer.buffer_size() < self.replay_mini_batch:
            return 0
        
        x, y = self._compute_regressors_targets()
        history = self.policy_nn.train_on_batch(x, y)
        return history.item()

    def _pretrain(self) -> None:
        """
        Routine devoted to pre-train our agent in some corner cases, which
        should improve network stability
        """
        # We'll pre_train the networks with selling all shares at the beginning of the interval
        # and selling all shares just at the end of the interval. Both of them are deemed very bad
        # practices, but will the data say so?

        # If the pre-train has gone well, we should see that the chances of our policy network picking
        # selling all shares at beginning is very low. On the other side, holding all shares
        # until the last episode should also be hugely penalized, considering that the reward function
        # is such that
        if self.pretrain:
            state = np.array([[self.env.q0, self.env.T]])
            x = self._compute_input(states=state)
            policy_start = self.policy_nn.predict(x[:, 1:], verbose=0)

            mode = 'start'
            history = []
            for pre_ep in range(self.pretrain):
                done = False
                ep_reward = 0
                loss = 0
                s,_ = self.env.reset()
                while not done:
                    action = self.choose_action(state=s, curr_greedy=0.01, train=True, pretrain_mod=mode)
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
                    history.append([s, action, reward, s_prime, self.env.data_idx])
                    # Update target (if needed)
                    self._soft_update_policy(pre_ep)


                # Change the pre-training mode
                if mode == 'start':
                    mode = 'end'
                else:
                    mode = 'start'

            state = np.array([[self.env.q0, self.env.T]])
            x = self._compute_input(states=state)
            policy_end = self.policy_nn.predict(x[:, 1:], verbose=0)
            return policy_start, policy_end
        
        return None,None
    
    def learn(self) -> list[float]:
        """
        Main function of the class devoted to train our agent with the data
        """
        policy_start, policy_end = self._pretrain()
        tot_ep_rewards = []
        tot_ep_losses = []
        history = []
        n = self.episodes-1 if not self.greedy_max_step else self.greedy_max_step
        step = 0
        for ep in range(self.episodes):
            s,_ = self.env.reset()
            done = False
            ep_reward = 0
            loss = 0
            while not done:
                greedy_param = max(((n-step)/n) *(self.greedy[0]-self.greedy[1]) + self.greedy[1],self.greedy[1])
                action = self.choose_action(
                    state=s, curr_greedy=greedy_param, train=True)
                s_prime, reward, done, _, info = self.env.step(action)
                self.replay_buffer.push(
                    s, action, reward, s_prime, done)
                ep_reward += reward
                loss += self._agent_update()
                if self.add_log:
                    history.append([s, action, reward, s_prime])
                s = s_prime
                self._soft_update_policy(ep)
                step +=1
            
            tot_ep_rewards.append(ep_reward)
            tot_ep_losses.append(loss)
            print(f'episode {ep}/{self.episodes-1}, greedy_param={round(greedy_param,3)}'\
                  f' reward={ep_reward}, avg_reward={round(sum(tot_ep_rewards)/len(tot_ep_rewards),3)}')

        return tot_ep_rewards, tot_ep_losses, history, policy_start, policy_end

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

    def _compute_targets(self, states, actions ,rewards, next_states, dones):
        """
        Auxiliary function devoted to compute the y-values for the networks
        """
        # max_q_s_prime_a_prime will be the max all predictions of policy network
        # over states s'. Note that in general, q_values are just the output values
        # for a given input state s over all actions that can be taken
        # Note that we are computing the q values from our target_nn, not our policy_nn
        if self.action_as_in:
            # TO_DO IMPORTANT!!! FIX
            x_next_states = self._compute_input(next_states)
            q_val = self.target_nn.predict(x_next_states[:, 1:], verbose=0)
            max_q_val = np.array([q_val[x_next_states[:, 0] == i].max()
                              for i, _ in groupby(x_next_states[:, 0])]).reshape(len(states), 1)
            y = rewards[:, 0] + self.gamma * \
            (states[:, 1] > self.env.dt).astype(int)*max_q_val[:, 0]
            return y

        target = self.policy_nn(tf.convert_to_tensor(states))
        target_next_states = self.policy_nn(tf.convert_to_tensor(next_states))
        next_state_val = np.array(self.target_nn(tf.convert_to_tensor(next_states)))
        
        max_action = np.argmax(target_next_states, axis=1)
        batch_index = np.arange(self.replay_mini_batch)
        y = np.copy(target)
        y[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index,max_action]*(1-dones.astype(int))

        return tf.convert_to_tensor(y)

    def choose_action(self, curr_greedy: float, state: np.array, train: bool, pretrain_mod: str = '') -> float:
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

        match pretrain_mod:
            case '':
                if train and np.random.rand() <= curr_greedy:
                    # Exploration
                    # Get a random trial from a binomial (qt , 1/time_to_expiry)
                    # Why are we enforcing on the exploration phase a TWAP approach? Should
                    # not the algorithm figure out that selling a lot of shares on an early
                    # point is not the best. Seems to me this is a way to reduce the exploration
                    # phase substantially => try with letting the algo pick any qt available
                    if self.unif:
                        action = np.random.randint(low=self.env.action_space.start,high=self.env.action_space.n)
                        return action
                    
                    qt, t = state[0, :]
                    return self.custom_exploration(qt,t, self.env.dt)
                            

                # Exploitation
                # Compute the q-value from all the possible actions [0,qt] and retrieve the action
                # that delivers the best overall q-value
                if self.action_as_in:
                    x = self._compute_input(states=state)
                    # Remember that the first element of x is just a mute index
                    predictions = self.policy_nn(x[:, 1:])
                else:
                    predictions = self.policy_nn(tf.convert_to_tensor(state.reshape(1,len(state))))
                best_action = np.argmax(predictions)
                return best_action
            case 'start':
                if t == self.env.T:
                    return self.env.q0
                return 0
            case 'end':
                if t == self.env.dt:
                    return self.env.q0
                return 0
            
# Create the ddqn agent of tensorflow
def initialize_ddqn_tf() -> DdqnAgent:
    dense_layers_ax = [create_dense_layer(neurons=neuron) for neuron in network_architecture['neurons']]
    dense_layers_ax.append(create_dense_layer(neurons=env.action_space.n,activation=linear))
    q_net_ax = tf_agents.networks.sequential.Sequential(dense_layers_ax)

    ddqn_tf = DdqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net_ax,
        optimizer=network_architecture['optimizer'],
        td_errors_loss_fn=network_architecture['loss_function'],
        target_update_period=agent_settings['nn_copy_cadency'])
    
    return ddqn_tf

def custom_exploration(qt,t,dt): 
    return binom(qt, dt/t).rvs()

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

# Create the gym environments and the agents
env = gym.make("CartPole-v0")
train_env = tf_py_environment.TFPyEnvironment(suite_gym.load('CartPole-v0'))

# SETTINGS
network_architecture = {'neurons': [128]*2,  # same params as DDQN paper
                        'output_size': env.action_space.n,
                        'loss_function': tf.keras.losses.MeanSquaredError(),
                        # modified according to DDQN paper
                        'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001),
                        }

agent_settings = {'gamma': 0.99,
                  'greedy': [1.0, 0.01],
                  'greedy_uniform': True,
                  'greedy_max_step': 500,
                  'environment': env,
                  'episodes': 1_000,  # 10_000 it's the param in DDQN
                  'buff_size': 500,  # 5_000 it's the param in DDQN
                  'replay_mini_batch': 64,  # 32 is the value used in DDQN, and seems the most generic one
                  'nn_copy_cadency': 10,  # every how many episodes we copy policy_nn to target_nn
                  'nn_architecture': network_architecture,
                  'soft_update': 0.005,
                  'pretrain': 0,
                  'action_as_input':False,
                  'cust_exploration':custom_exploration,
                  'add_logs':False}

# In order to make different runs reproducible, fix random seeds
seed = 5
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

ddqn_axel = DDQN(sett=agent_settings)
ddqn_tf = initialize_ddqn_tf()

# Copy params from 
# https://github.com/abhisheksuran/Reinforcement_Learning/blob/master/DDDQN.ipynb

# According to openai/gym/wiki
# Considered solved when the average reward is greater than 
# or equal to 475 over 100 consecutive episodes.

#Train ddqn implementation on the Cartpole environment
t1 = time.time()
rewards, losses, logs, pol_start, pol_end = ddqn_axel.learn()
t2 = time.time()
print(f'Time learning ddqn agent with numpy arrays={round(t2-t1,3)}')

plot(data=rewards, title='rewards vs 15 period mavg', mavg=False)
plot(data=losses, title='losses vs 15 period mavg', mavg=False)

# Things to check at the moment
# Around episode 60ish for Cartpole-v0 everything seems
# to explode, check where we are being so inneficient

