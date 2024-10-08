from collections import deque
from itertools import groupby
import gymnasium as gym
import numpy as np
import random
from scipy.stats import binom
import sys
sys.path.insert(1,"/home/axelbm23/Code/ML_AI/Algos/Miscellaneous/")
from misc_utils import create_nn
import tensorflow as tf
from typing import Tuple
import pickle

class ExpirienceReplay:
    """
    Deque object parametrized by two settings: The max length
    of the dequeu (maxlen) and the length of the elements
    that are retrieved when user pushes data (mini_batch).
    This object is developed with the aim to be used as a Replay
    Buffer for Reinforcement Learning agents.
    """

    def __init__(self, maxlen: int, mini_batch: int):
        self._buffer = deque(maxlen=maxlen)
        self.mini_batch = mini_batch

    def push(self, state, action, reward, next_state, terminated):
        if state.ndim==1:
            state = state.reshape(1,len(state))
            next_state = next_state.reshape(1,len(next_state))
        
        self._buffer.append(
            (state, action, reward, next_state, terminated))

    def _get_batch(self):
        return random.sample(self._buffer, self.mini_batch)

    def get_arrays_from_batch(self):
        batch = self._get_batch()

        states = np.array([x[0] for x in batch])[:, 0, :]
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])[:, 0, :]
        terminated = np.array([x[4] for x in batch])

        return states, actions, rewards, next_states, terminated

    def buffer_size(self):
        return len(self._buffer)
    
class DDQN:
    """
    Object implementing a DDQN (double deep Q neuronal network)
    algorithm to be used as agent in any RL environment
    """

    def __init__(self,
                 sett: dict) -> None:

        self.gamma = sett['gamma']
        self.greedy = 1.0
        self.greedy_step = sett['greedy_step']
        self.env = sett['environment']
        self.episodes = sett['episodes']
        self.buff_size = sett['buff_size']
        self.replay_mini_batch = sett['replay_mini_batch']
        self.replay_buffer = ExpirienceReplay(
            maxlen=self.buff_size, mini_batch=self.replay_mini_batch)
        self.batch_index = np.arange(self.replay_mini_batch)
        self.action_as_in = sett['nn_architecture']['action_as_input']
        self.policy_nn = create_nn(input_size=self.env.observation_space.shape[0]+1 if self.action_as_in 
                                   else self.env.observation_space.shape[0],
                                    model_params=sett['nn_architecture'],
                                    output_size=1 if self.action_as_in else self.env.action_space.n)
        self.target_nn = create_nn(input_size=self.env.observation_space.shape[0]+1 if self.action_as_in 
                                   else self.env.observation_space.shape[0],
                                    model_params=sett['nn_architecture'],
                                    output_size=1 if self.action_as_in else self.env.action_space.n)
        
        self.target_nn_initialized = True
        self.already_copied = False
        self.copy_cadency = sett['nn_copy_cadency']
        self.soft_update = sett['soft_update']
        self.add_log = sett['add_logs']
        self.solve_metric = sett['solve_metric']
        self.solve_target = sett['solve_target']
        self.termination_policy = self.solve_metric != None
        self.max_action = sett['max_poss_action']
        self.man_grad = sett['man_gradient']

    def _soft_update_policy(self, ep:int) -> None:
        """
        Function that specifies the policy used to copy
        the weights of the policy network to the target
        network.
        """

        old_weights = self.target_nn.get_weights()
        new_weights = self.policy_nn.get_weights()
        if self.target_nn_initialized:
            if self.add_log:
                print(f'Hard copy policy_weights to target_weights')
            self.target_nn.set_weights(new_weights)
            self.target_nn_initialized = False
            return 
        
        if self.copy_cadency:
            if (ep+1) % self.copy_cadency == 0.0:
                if not self.already_copied:
                    if self.add_log:
                        print(f'Hard copy policy_weights to target_weights')
                    self.target_nn.set_weights(new_weights)
                    self.already_copied=True
                return
            
            self.already_copied = False
            return
        
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

        x = tf.convert_to_tensor(states)
        if self.action_as_in:
            x = tf.convert_to_tensor(self._compute_input(states)[:,1:])
        y = self._compute_targets(
            states, actions, rewards, next_states, dones)

        # Perform dimensionality check
        if x.shape[1]!=self.target_nn.input_shape[1] or y.ndim!=self.target_nn.output_shape[1] or x.shape[0]!=y.shape[0]:
            raise ValueError(f'x-y dimensions are wrong, check!')
        return x, y
    
    def _agent_update(self) -> float:
        """
        Main function devoted to update params of the policy network
        """
        if self.replay_buffer.buffer_size() < self.replay_mini_batch:
            return 0
        
        x, y = self._compute_regressors_targets()
        
        if self.man_grad:
            with tf.GradientTape() as tape:
                y_pred = self.policy_nn(x, training=True)
                # Compared loss with tf.keras.losses.MeanSquaredError() and outputs the same
                # So discrepancy has to be on another side
                loss = self.policy_nn.compute_loss(y=y, y_pred=y_pred)
            
            
            trainable_vars = self.policy_nn.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            iters = self.policy_nn.optimizer.apply_gradients(zip(gradients, trainable_vars))
        else:
            history = self.policy_nn.fit(x=x, y=y, verbose=0, epochs=1, batch_size=self.replay_mini_batch)
            loss = history.history['loss'][0]

        # Compute the gradient of my loss function wrt model weights
        # Basically to ensure that there is no gradient explosion
        
        return loss

    def train(self) -> list[float]:
        """
        Main function of the class devoted to train our agent with the data
        action_chooser is passed as a parameter to allow flexibility over
        different agents, which may have different ways to chose the action
        but can benefit from the same code to train. 
        """
        tot_ep_rewards = []
        tot_ep_losses = []
        history = []
        step = 0
        for ep in range(self.episodes):
            s,_ = self.env.reset()
            done = False
            ep_reward = 0
            loss = 0
            while not done:
                action = self.choose_action(state=s)
                s_prime, reward, done, _, info = self.env.step(action)
                self.replay_buffer.push(s, action, reward, s_prime, done)
                if self.add_log:
                    history.append([s, action, reward, s_prime, info['timestamp']])
                self._soft_update_policy(ep)
                ep_reward += reward
                loss += self._agent_update()
                s = s_prime
                step += 1
                self.greedy *= self.greedy_step
            
            tot_ep_rewards.append(ep_reward)
            tot_ep_losses.append(loss)
            avg = np.average(tot_ep_rewards)
            last_hundred_avg =  np.average(tot_ep_rewards[-100:])
            if self.add_log:
                print(f'episode {ep}/{self.episodes-1}, greedy_param={round(self.greedy,5)}'\
                    f' reward={ep_reward}, avg_rew={round(avg,3)},'
                    f' avg_rew(100)={round(last_hundred_avg,3)},'
                    f' loss={loss}')
            
            if self.termination_policy and self.solve_metric(tot_ep_rewards) >= self.solve_target:
                print(f'Success, agent has solved the environment')
                break
        
        tot_ep_rewards = np.array(tot_ep_rewards)
        tot_ep_losses = np.array(tot_ep_losses)
        
        return tot_ep_rewards, tot_ep_losses, history

    def _compute_input(self, states: np.array, actions: np.array = np.array([]), action_flag : np.array = np.array([])) -> np.array:
        """
        This function creates the input needed for the nn, both for the
        target and for the policy. If no actions are provided, it generates
        all the possible actions (0, qt) for the current state.

        states => MUST BE a ndim=2 np.array
        actions => MUST BE a ndim=1 np.array
        """
        if actions.size == 0:
            x = np.array([])
            for idx, state in enumerate(states):
                if not self.max_action:
                    max_action_len = self.env.action_space.n-1
                else:
                    max_action_len = self.max_action(state[0])
                mod_state = np.append(
                    np.array([idx]), states[idx:idx+1, :]).reshape(1, state.shape[0]+1)
                rep_states = np.repeat(mod_state, max_action_len+1, axis=0)
                poss_actions = np.array(
                    range(max_action_len+1), dtype=int).reshape(max_action_len+1, 1)
                repeated_state = np.append(rep_states, poss_actions, axis=1)
                # Add a flag to know which was the action selected
                if action_flag.size != 0:
                    repeated_state = np.c_[repeated_state,repeated_state[:,-1]==action_flag[idx]]
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

        return np.append(states, actions.reshape(len(actions),1), axis=1)
    
    def _construct_np_arr(self, x, q_vals):
        """
        x: tensor containing the index of the state on the first column and the action
            taken on the last column
        q_vals : tensor result from computing q_vals on x

        returns a numpy array appending information from both tensors
        """
        return np.append(x.numpy()[:,[0,-1]], q_vals.numpy(),axis=1)

    def _compute_targets(self, states, actions ,rewards, next_states, dones):
        """
        Auxiliary function devoted to compute the y-values for the networks
        """
        
        if self.action_as_in:
            # Exercise => rewrite this so that action is taken as input
            # Consider that Y needs to be a self.batch x 1d array
            x_states = tf.convert_to_tensor(self._compute_input(states, action_flag=actions))
            x_next_states = tf.convert_to_tensor(self._compute_input(next_states))

            # Perfom some checks
            if sum(x_states.numpy()[:,-1]==1)!=self.replay_mini_batch:
                raise ValueError(f'Check x_states computation, number of selections actions does not match!!')

            target = self._construct_np_arr(x_states, self.policy_nn(x_states[:,1:-1]))
            predicted_states = self.policy_nn(x_next_states[:,1:])

            # All the Nans come from the fact that policy_nn predicts all nans, this seems a policy gradient explosion
            # For instance, I am seeing values of 6.50742775939072e+16 for the loss function
            if any(np.isnan(predicted_states)):
                print(f'target_next_states has Nans, check!!')
            target_next_states = self._construct_np_arr(x_next_states, predicted_states)
            
            # One corner case here is what happens if the policy_nn network produces the exact same q_value
            # for two diferent actions. In those cases it's indiferent which action to take, so we take
            # the first one (arbitrary, but by definition should not have any impact). Also,
            # we expect this to vanish as soon as the networks starts learning.
            max_actions = np.array([target_next_states[((target_next_states[:,0]==i)&(target_next_states[:,2]==target_next_states[target_next_states[:,0]==i,2].max())),1][0]\
                          for i, _ in groupby(target_next_states[:, 0])])
            
            x_next_states_with_max_action = tf.convert_to_tensor(self._compute_input(next_states,max_actions))
            max_q_prime = self.target_nn(x_next_states_with_max_action)
            
            
            y = np.copy(target)
            y[x_states[:,-1]==1, 2] = rewards + self.gamma*max_q_prime[:,0]*(1-dones.astype(int))
            return tf.convert_to_tensor(y[:,2])

        states_ts = tf.convert_to_tensor(states)
        next_states_ts = tf.convert_to_tensor(next_states)

        target = self.policy_nn(states_ts)
        # This will output a [self.batch x 2] array
        target_next_states = self.policy_nn(next_states_ts)
        next_state_val = np.array(self.target_nn(next_states_ts))
        max_actions = np.argmax(target_next_states, axis=1)
        
        y = np.copy(target)
        y[self.batch_index, actions] = rewards + self.gamma * next_state_val[self.batch_index,max_actions]*(1-dones.astype(int))

        return tf.convert_to_tensor(y)

    def choose_action(self, state: np.array, train: bool = True) -> float:
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
        
        if train and np.random.rand() <= self.greedy:
            # Exploration
            # Get a random trial from a binomial (qt , 1/time_to_expiry)
            # Why are we enforcing on the exploration phase a TWAP approach? Should
            # not the algorithm figure out that selling a lot of shares on an early
            # point is not the best. Seems to me this is a way to reduce the exploration
            # phase substantially => try with letting the algo pick any qt available
            action = np.random.randint(low=self.env.action_space.start,high=self.env.action_space.n)
            return action

        # Exploitation
        # Compute the q-value from all the possible actions [0,qt] and retrieve the action
        # that delivers the best overall q-value
        if self.action_as_in:
            x = self._compute_input(states=state.reshape(1,len(state)))
            # Remember that the first element of x is just a mute index
            predictions = self.policy_nn(tf.convert_to_tensor(x[:, 1:]))
        else:
            predictions = self.policy_nn(tf.convert_to_tensor(state.reshape(1,len(state))))
        
        best_action = np.argmax(predictions)
        return best_action
    
    def test(self, env:gym.Env, episodes:int) -> Tuple[np.array, list]:
        logs = []
        rew = []
        # Make sure env is always set to the start
        s,_ = env.reset(0)
        for ep in range(episodes):
            ep_rew = 0
            done = False
            while not done:
                action = self.choose_action(state=s, train=False)# Issue seems that done condition is not satisfied
                s_prime, reward, done, _, info = env.step(action)
                logs.append([s, action, reward, s_prime, info['timestamp']])
                s = s_prime
                ep_rew += reward

            rew.append(ep_rew)
            # Reset environment as episode has concluded, so inventory and
            # rest of quantities need to be reset. Do not reset at index 0
            # though, as we want to run on different data
            s,_ = env.reset()
        rew = np.array(rew)

        return rew, logs
    
    def save(self, path:str, id:str) -> None:
        """
        Saves current state of the agent in path folder with the name id.
        """

        # Generate unique 6 character identifier for this agent
        pol_nn_id = f'{id}_policy.keras'
        tgt_nn_id = f'{id}_target.keras'
        sett_id = f'{id}_sett.pkl'

        self.policy_nn.save(f'{path}/{pol_nn_id}')
        self.target_nn.save(f'{path}/{tgt_nn_id}')
        
        # Save rest of settings into a pkl object
        sett = dict((key,value) for key,value in self.__dict__ .items() if key not in (['policy_nn','target_nn']))
        with open(f'{path}/{sett_id}', 'wb') as f:
            pickle.dump(sett, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        pass
    
    def load(self, path:str, id:str):
        """
        Loads models and all parameters related to the agent in path with
        identifier id
        """

        file = open(f'{path}/{id}_sett.pkl', 'rb') 
        for key,value in pickle.load(file).items():
            self.key = value

        self.policy_nn = tf.keras.models.load_model(f'{path}/{id}_policy.keras')
        self.target_nn = tf.keras.models.load_model(f'{path}/{id}_target.keras')

        return self

class DDQN_tradexecution(DDQN):
        """
        Child class of DDQN with custom logic to select actions and pretrain
        the model. Both custom functions affect the training of the agent.
        """
        def __init__(self,sett):
            super(DDQN_tradexecution, self).__init__(sett)
            self.unif = sett['greedy_uniform']
            self.pretrain = sett['pretrain']

        def choose_action(self, state: np.array, train: bool = True, pretrain_mod: str = '') -> float:
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
            
            qt, t = state
            match pretrain_mod:
                case '':
                    if train and np.random.rand() <= self.greedy:
                        # Exploration
                        # Get a random trial from a binomial (qt , 1/time_to_expiry)
                        # Why are we enforcing on the exploration phase a TWAP approach? Should
                        # not the algorithm figure out that selling a lot of shares on an early
                        # point is not the best. Seems to me this is a way to reduce the exploration
                        # phase substantially => try with letting the algo pick any qt available
                        if self.unif:
                            action = np.random.randint(low=self.env.action_space.start, high=qt+1)
                            return action
                        return binom(qt, self.env.dt/t).rvs()
                                

                    # Exploitation
                    # Compute the q-value from all the possible actions [0,qt] and retrieve the action
                    # that delivers the best overall q-value
                    if self.action_as_in:
                        x = self._compute_input(states=state.reshape(1,len(state)))
                        # Remember that the first element of x is just a mute index
                        predictions = self.policy_nn(tf.convert_to_tensor(x[:, 1:]))
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
                        action = self.choose_action(state=s, pretrain_mod=mode)
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
                        history.append([s, action, reward, s_prime, info['timestamp']])
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
        
        def train(self):
            # If we pre-train we are actually using episodes+pre_train episodes
            # to train the model, so more data than if we don not pre-train it
            pol_start, pol_end = self._pretrain()
            
            # Since we have modified the choose_action to our new needs,
            # we can just call the parent class with train
            tot_ep_rewards, tot_ep_losses, history = super(DDQN_tradexecution, self).train()
            return tot_ep_rewards, tot_ep_losses, history, pol_start, pol_end
        
class TWAP(DDQN):
    """
    Very simple agent that does not learn anything, just
    replicates behaviour of a TWAP strategy
    """
    def __init__(self,sett):
        super(TWAP, self).__init__(sett)

    # We need to overwrite _agent_update and
    # _soft_update_policy as they are called
    # inside DDQN.train method
    def _agent_update(self) -> int:
        return 0
    
    # TO-DO: Substitute this with optional arguments
    def _soft_update_policy(self, ep:int) -> None:
        pass

    def choose_action(self, state: np.array, train: bool = True):
        return int(self.env.q0/self.env.N)
    
class RANDOM_TE(TWAP):
    """
    Random agent that executes a random number of shares
    on each time step
    """
    def __init__(self,sett):
        super(RANDOM_TE, self).__init__(sett)
    
    # TO-DO: Substitute this with optional arguments
    def choose_action(self, state: np.array, train: bool = True):
        return np.random.randint(0, state[0]+1)

class Agent_Performance():
    def __init__(self, rewards:list[np.array], losses:list[np.array], solved_func, solved_rew:float) -> None:
        """
        rewards : list of variable numpy arrays, where each array contains the rewards for a given trial.
        losses : list of variable size numpy arrays, where each array contains the losses for a given trial.
        solved_func : function that takes a list of rewards as input and returns the statistic need to
        know if the env is solved.
        solved_rew : if solved_func==solved_rew, then env is solved
        """
        self.solv_cond = solved_func
        self.solved_rew = solved_rew
        self.rew = rewards
        self.loss = losses
        self.rew_return = [np.diff(trial) for trial in self.rew]

    def _avg_end_rew(self) -> float:
        """
        Computes average end reward for each trial
        """
        return np.average([trial[-1] for trial in self.rew])
    
    def _best_worst_end_rew(self) -> float:
        """
        Ratio between best and worst try
        """
        best_end_rew = max([trial[-1] for trial in self.rew])
        worst_end_rew = min([trial[-1] for trial in self.rew])

        return best_end_rew/worst_end_rew - 1
    
    def _avg_across_trials(self,qty) -> float:
        """
        Given a qty, it computes the average of that qty across trials
        """
        total_points = sum([trial.size for trial in qty])
        total_rewards = sum([sum(trial) for trial in qty])

        return total_rewards/total_points

    def _avg_epi_rew(self) -> float:
        """
        Computes avg reward per episode
        """
        return self._avg_across_trials(self.rew)
    
    def _avg_epi_rew_increase(self) -> float:
        """
        Computes avg reward increase per episode
        """ 
        return self._avg_across_trials(self.rew_return)
    
    def _std_epi_rew_increase(self) -> float:
        """
        Computes the standard deviation of the reward increase per episode
        """
        return np.mean([np.var(trial) for trial in self.rew_return])**0.5
    
    def _n_solved(self) -> int:
        """
        Computes number of trials that were solved
        """
        return sum([self.solv_cond(trial)>=self.solved_rew for trial in self.rew])
    
    def _max_drawdown(self) -> float:
        """
        Computes average drawdown in the learning process, as the difference between
        the max value and the subsequent min value. This aims to capture the agent
        de-learning the right policy.
        """
        min_max = []
        for trial in self.rew:
            trial_min = min(trial[np.argmax(trial):])
            min_max.append(max(trial)-trial_min)

        return np.average(min_max)
    
    def _avg_ep_to_solve(self) -> float:
        ep_solv = []
        for trial in self.rew:
            if self.solv_cond(trial)>=self.solved_rew:
                # It assumes that env stops when it is solved
                ep_solv.append(len(trial))
        
        return None if ep_solv==[] else np.average(ep_solv)

    def compute_statistics(self, prec:int) -> dict:
        return {'avg_end_rew':round(self._avg_end_rew(),prec), # info about overall learning capacity
                'best_worst_ratio_end_rew':round(self._best_worst_end_rew(),prec),# robustness across best-worst try
                'avg_epi_rew':round(self._avg_epi_rew(),prec), # info about the episode learning capacity
                'avg_epi_rew_increase':round(self._avg_epi_rew_increase(),prec), # info about the learning rate
                'std_epi_rew_increase':round(self._std_epi_rew_increase(),prec), # info about the robustness on the learning
                'n_solved':self._n_solved(),# info about overall success in the learning
                'avg_ep_to_solve':self._avg_ep_to_solve(),# info about how fast agent solves it
                'max_drawdown':round(self._max_drawdown(),prec) # info about how likely is that agent to de-learn the optimal policy 
                }