import datetime
from gymnasium import spaces
from gymnasium.core import RenderFrame
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Any
from typing import SupportsFloat
# sys.path.insert(2, '/home/axelbm23/Code/AlgoTrading/data/')
# sys.path.insert(3, '/home/axelbm23/Code/ML_AI/Algos/Utils')


def load_data(path: str, data_points: int = 0) -> pd.DataFrame:
    """
    Function devoted to import the data and format it accordingly
    """

    # col_labels = ['Index', 'Timestamp', 'Datetime']
    # side_labels = [f'{side}{att}{i}' for side in ['b','a'] for i in range(10) for att in ['p','q']]
    col_labels = ['Timestamp', 'bp0', 'bq0', 'ap0', 'aq0']
    df = pd.read_csv(path, nrows=data_points if data_points > 0 else None, usecols=[1, 3, 4, 23, 24],
                     names=col_labels, header=0)

    # Convert the Timestamp into a datetime
    df['datetime'] = df['Timestamp'].apply(
        lambda x: datetime.datetime.fromtimestamp(x/1e3))
    df['mp'] = 0.5*(df['bp0']+df['ap0'])

    # Return only the needed columns
    return df[['datetime', 'mp']]


def format_history(hist: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(hist, columns=['qt', 't', 'reward', 'algo', 'episode'])


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

    def reset(self, seed=None) -> tuple[Any, dict[str, Any]]:
        self.qt = self.q0
        self.t = self.T
        self.data_idx = 0
        self.done = False

        info = {'Inventory': self.qt,
                'Time_to_execute': self.t}

        return self.extract_state(), info

    # IMPORTANT => all columns that define a state should start with prefix state_
    def extract_state(self) -> tuple[int, int]:
        return (self.qt, self.t)

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
        shs_per_interval = self.action/self.mk
        if shs_per_interval % 1.0 != 0.0:
            raise ValueError(
                f'shs_per_interval is not an int, {shs_per_interval}')

        for _ in range(self.mk):
            curr_price = self._get_mid_price(self.data_idx)
            next_price = self._get_mid_price(self.data_idx+1)
            reward += self.qt*(next_price-curr_price) - \
                self.alpha*shs_per_interval**2
            self.data_idx += 1
            self.qt -= shs_per_interval

        # Update time left to execute
        self.t -= self.dt

        # Distribute some printing information
        info = {'Inventory': self.qt,
                'Time_to_execute': self.t}

        # The terminal condition is satisfied when we have no time left
        # to close our position

        # TO-DO: How do we promote that at the last step, the agent needs to get rid
        # of all current inventory?? This should be enforced in the agent, not at
        # the environment level.

        # If our inventory gets to 0 or we don't have more time to unfold our position,
        # we consider the episode to be finished
        if self.t == 0 or self.qt == 0:
            self.done = True

        # Note here observation needs to be the state s' which we have
        # arrived after applying action to state s
        obs = self.extract_state()

        # the result of step needs to be observation, reward, done, info
        return obs, reward, self.done, False, info


path = "/home/axelbm23/Code/AlgoTrading/RL/TradeExecution/data/1-09-1-20.csv"
df = load_data(path)

# Create a trade execution environment with some random settings.
te_env = TradingEnv(data=df, T=3600, q0=14_400, N=8)

# Just for testing purposes, we will implement an
# agent that follows a TWAP strategy.
episodes = 100
history = []
twap_action = int(te_env.q0/te_env.N)
for trie in range(episodes):
    te_env.reset()
    done = False
    while not done:
        qt, t = te_env.extract_state()
        obs, reward, done, _, info = te_env.step(twap_action)
        history.append(obs+(reward, 'twap', trie))

# TO_DO: What if the algo decides that the amount of shares to be traded
# in a given time interval is not an integer value

results = format_history(history)
