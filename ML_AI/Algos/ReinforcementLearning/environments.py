import gymnasium as gym
from gymnasium.core import RenderFrame
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Any
from typing import SupportsFloat

class TradeExecution(gym.Env):
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
    

class PortfolioOptimization(gym.Env):
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

    def __init__(self, data: pd.DataFrame, trans_costs: float, slipp: float,
                 trade_size: int, initial_balance: float, long_only: bool):
        self.data = data
        # Here we assume a constant risk free rate specified
        # by the user over the entire interval
        # TO_DO: Use T10 treasury yields as a more realistic proxy for this
        # as its value changes every day
        self.action = 0
        self.position = 0
        self.curr_index = 0
        self.max_trade_size = trade_size
        self.initial_balance = initial_balance
        self.portfolio_value = initial_balance
        self.terminal_idx = self.data.shape[0] - 1
        self.long_only = long_only
        self.tc = trans_costs
        self.slippage = slipp

        # This is the variable that controls whether we have reached a
        # terminal state, which can be triggered in one of the two
        # scenarios:
        #   - We've reached the end of the training set
        #   - Our balance turns negative.
        self.done = False

        # This is where we define what possible actions can the agent take
        # THIS NEEDS TO BE TAILORED TO THE ACTIONS WE ALLOW THE AGENT TO TAKE
        # MANDATORY TO DEFINE THIS ARGUMENT IN OUR gym class.
        # COMMENT: IN order to avoid issues with the agent taking actions
        # outside the valid range defined by action_space (gym env
        # does not enforce the lower/upper limits that the agent can take),
        # we will define out action space as -1 to 1 and we will transform it
        # to shares considering
        self.action_space = spaces.Box(low=-1.0, high=1.0, dtype=float)

        # THIS NEEDS TO BE TAILORED TO WHAT WE DEFINE AS A STATE
        # MANDATORY TO DEFINE THIS IN OUR gym class.
        # In general, we define as a state everything that agent
        # can observe and that will use to take a given action.

        # In our data, the state space will be defined by:
        #   - Position in our stock
        #   - Balance at current time step t
        #   - Adjusted closed price of the stock
        #   - MACD
        #   - RSI
        #   - Commodity Channel Index (CCI)

        # We will define the limits of each component of our observation space
        # independently. Note that our position can not go negative, so we do
        # not allow short selling

        # TO_DO: Generalize this so that the the observation space can be dynamically created
        # once the user has created a set of observations (allow the user to dynamically create
        # a set of predictors)
        self.observation_space = spaces.Box(low=np.array([0, 0, limits['stock_price'][0],
                                                          limits['MACD'][0], limits['RSI'][0],
                                                          limits['CCI'][0]]),
                                            high=np.array([np.inf, np.inf,
                                                           limits['stock_price'][1], limits['MACD'][1],
                                                           limits['RSI'][1], limits['RSI'][1]]),
                                            dtype=np.float64)

    def reset(self, seed=None) -> tuple[Any, dict[str, Any]]:
        self.position = 0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.curr_index = 0
        self.action = 0

        info = {'date': self.data.index[self.curr_index],
                'portfolio_value': self.portfolio_value,
                'PnL': 0.0,
                'portfolio_return': 0.0,
                'asset_return': 0.0,
                'position': self.position,
                'traded_price': self.data.iloc[self.curr_index]['Close'],
                'ref_price': self.data.iloc[self.curr_index]['Close'],
                'action': self.action,
                'shares_traded': 0}

        return self.extract_state(), info

    # IMPORTANT => all columns that define a state should start with prefix state_
    def extract_state(self) -> pd.Series:
        state_vars = [col for col in self.data if col.startswith('state_')]
        indicators = self.data.iloc[self.curr_index][state_vars]
        # On top of the above columns we need to retrieve
        # the current position and the available balance as they are both
        # state variables
        portf_vars = pd.Series(
            [self.position, self.balance], index=[self.curr_index]*2)

        return pd.concat([portf_vars, indicators])

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:

        # This is where we define of how the environment reacts
        # when agent takes a given action.

        # The first thing we will do is to asses whether the provided action is
        # a valid member of the action space defined by self.action_space
        assert self.action_space.contains(action)

        # For now we will compute our reward as a function
        # of the change in the portfolio value
        prev_portfolio_value = self.portfolio_value
        prev_price = self.data.iloc[max(self.curr_index-1, 0), :]['Close']
        current_price = self.data.iloc[self.curr_index, :]['Close']
        sign = -1 if action[0] < 0 else 1
        traded_price = current_price*(1+sign*self.slippage)
        shares_traded = int(action[0]*self.max_trade_size/traded_price)

        # QUESTION: SHOULD NOT THE REWARD ONLY TAKE INTO ACCOUNT THE
        # CHANGE IN THE PORTFOLIO VALUE before actually executing
        # the action?!

        # We are selling, our balance increases
        # If we are buying, our balance decreases
        # If we hold position, our balance does not change

        # Impose restriction in case we do not allow short-selling
        if self.long_only and self.position+shares_traded < 0:
            shares_traded = -1*self.position

        self.balance += -shares_traded*traded_price
        self.position += shares_traded
        curr_portfolio_value = self.balance + self.position*current_price
        comission_cost = self.tc*abs(shares_traded)*traded_price
        slippage_cost = shares_traded*(traded_price-current_price)

        # This qty>=0 always!!!
        trans_cost = comission_cost+slippage_cost

        # Note that the change in the portfolio value will be associated with the change
        # of monetary value - cost_of_transaction - slippage_costs. Adding transaction_costs
        # and slippage costs not only translates the model to a more realistic one,
        # but also ensures that the agent gets penalized for trading very often!!!
        reward = curr_portfolio_value - prev_portfolio_value - trans_cost

        # Distribute some printing information
        info = {'date': self.data.index[self.curr_index],
                'portfolio_value': curr_portfolio_value, 'PnL': reward,
                'portfolio_return': curr_portfolio_value/self.portfolio_value - 1,
                'asset_return': current_price/prev_price - 1,
                'position': self.position,
                'traded_price': traded_price,
                'ref_price': current_price,
                'action': action,
                'shares_traded': shares_traded}

        # Make all quantities advance
        self.portfolio_value = curr_portfolio_value
        self.action = action
        # Make the time advance
        self.curr_index += 1

        # The terminal conditions are either we arriving to the end
        # of the simulation or our balance being negative
        if self.curr_index == self.terminal_idx:
            self.done = True

        # Note here observation needs to be the state s' which we have
        # arrived after applying action to state s
        obs = self.extract_state()

        # the result of step needs to be observation, reward, done, info
        return obs, reward, self.done, False, info
