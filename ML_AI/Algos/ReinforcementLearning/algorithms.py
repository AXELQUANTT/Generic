"""
Package with some implementations of RL algos/agents
"""

from Utils import create_nn
from alphas_utils import limits
from typing import Any
from typing import SupportsFloat
from tf_agents.replay_buffers import py_uniform_replay_buffer
import sys
import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame
import gymnasium as gym
import pandas as pd
import keras
sys.path.insert(1, '/home/axelbm23/Code/AlgoTrading/alphas/')
sys.path.insert(2, '/home/axelbm23/Code/AlgoTrading/data/')
sys.path.insert(3, '/home/axelbm23/Code/ML_AI/Algos/Utils')


class TradingEnv(gym.Env):
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
