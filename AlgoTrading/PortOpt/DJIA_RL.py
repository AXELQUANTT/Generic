"""
Project designed to replicate a a deep-Q reinforcement
learning algorithm for stock trading purposes.
"""

from Utils import TradingEnv
from RL_Algorithms import TradingEnv
from yahoo_loader import yf_ticker_loader
from gymnasium import register
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import sys
import ta
import ta.momentum
import ta.trend
sys.path.insert(1, '/home/axelbm23/Code/AlgoTrading/data/')
sys.path.insert(2, '/home/axelbm23/Code/ML_AI/Algos/')


def format_statistics(hist: list[dict], logs: dict) -> pd.DataFrame:
    statistics = pd.DataFrame(hist, columns=logs.keys())
    statistics.set_index('date', inplace=True)
    statistics['portf_cumreturn'] = (
        1+statistics['portfolio_return']).cumprod()
    statistics['asset_cumreturn'] = (1+statistics['asset_return']).cumprod()

    return statistics


def porfolio_asset_plot(stats: pd.DataFrame, ticker: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(stats['portf_cumreturn'], label='portfolio_cumreturn')
    ax.plot(stats['asset_cumreturn'], label=f'{ticker}_cumreturn')
    fig.autofmt_xdate()
    plt.legend()
    plt.plot()

# Steps
# 1. Download or query historical data for a set of stocks.
#   Choose a universe of stocks that is manageable, like DJIA
#   Index for instance => Done
# 2. Create a OpenAI gym environment for stock trading.
# 3. Run agent over defined period to train it.

# Since this is only for learning purposes, we will only
# consider a single stock in our portfolio.

# The next consideration is the number of actions
# the agent can take. For our purposes,
# the agent is only allowed to either buy, hold, or
# sell (in case we have a position). Therefore, we don't
# allow the algo to go short the stock.


# The last thing we will impose is the algorithm used
# by our agent to decide which action is the one that
# maximizes rewards. There are several algorithms
# used as agents.
ticker = 'MSFT'
df = yf_ticker_loader(ticker, '2020-01-01', '2021-12-31', True)

# Create a range of signals used for a posteriori analysis
df['state_Close'] = df['Close']
# We'll compute MACD with the usual settings for the crossover
# slow_avg_window = 26, fast_avg_window = 12, signal_window=9
df['state_MACD'] = ta.trend.macd(df['Close'])
# Rsi calculation also takes the default values
df['state_RSI'] = ta.momentum.rsi(df['Close'])
# The same for CCI index
df['state_CCI'] = ta.trend.cci(
    high=df['High'], low=df['Low'], close=df['Close'])

# Make sure that the dataframe we send to the environment does not contain any NaN
df.dropna(inplace=True)

max_trade_size = 10_000
initial_balance = 1_000_000
tradenv = TradingEnv(data=df, trans_costs=0.0,
                     slipp=0.0, trade_size=max_trade_size,
                     initial_balance=initial_balance,
                     long_only=True)

# Now that we have created the environment, we can test it with a
# very stupid agent which takes random actions
obs = tradenv.reset()
done = False
history = []
while not done:
    action = np.random.uniform(low=-1.0, high=1.0, size=1)
    obs, reward, done, _, info = tradenv.step(action)
    history.append(info)

statistics = format_statistics(history, info)
porfolio_asset_plot(statistics, ticker)

# Now let's train an agent
# In order to do this, we'll use some already created algorithms
# such as PPO.
model = PPO("MlpPolicy", tradenv, verbose=0)
model.learn(total_timesteps=100_000, progress_bar=True)

# Now what we can do is to try on the same dataset how our agent would
# have performed. We expect our agent to do REALLY good here, since
# it has been trained in this data
# Reset the environment and use our 'agent', a.k.a trader,
# to trade it
obs, _ = tradenv.reset()
ppo_history = []
for i in range(len(df)-1):
    action, _states = model.predict(obs)
    obs, rewards, done, _, info = tradenv.step(action)
    ppo_history.append(info)
ppo_stats = format_statistics(ppo_history, info)
porfolio_asset_plot(ppo_stats, ticker)
