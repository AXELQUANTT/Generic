"""
Script devoted to test and compare
multiple RL agents over a standard
discretization of the Ornstein-Uhlenbeck
"""


import pandas as pd
import numpy as np
import sys
#sys.path.insert(1, '/home/axelbm23/Code/ML_AI/Algos/')
#from RL_Algorithms import TradingEnv

def create_mean_reverting_signal(p_init: float, kappa: float, theta: float, sigma: float, size: int) -> pd.DataFrame:
    """
    This function creates an artificial time series defined as discretization
    of a Ornstein-Uhlenbeck process:
        - p_init: initial price
        - kappa: mean reversion coefficient
        - theta: long term mean price
        - sigmal: volatility coefficient

    """

    prices = [p_init]
    norm = np.random.normal(size=size)
    for idx in range(1, size):
        prices.append(theta+(prices[idx-1]-theta)*np.exp(-1*kappa) +
                      norm[idx]*np.sqrt((0.5*sigma**2/kappa)*(1.0-np.exp(-2*kappa))))

    return pd.DataFrame(prices, index=list(x for x in range(size)), columns=['signal'])

# Steps of the project
#   1.  Create the data
#   2.  Set up custom trading environment
#   3.  Train multiple agents on the data
#   4.  Asses the performance of multiple agents
#       and cross check them


# Data creation
data_size = 100_000
thet = 55.0  # this is mean price
p = thet*(1+0.05)
sig = 0.1  # how volatile the signal is around its mean price
kap = 0.01  # the bigger the value, the faster the signal mean reverts towards thet


signal = create_mean_reverting_signal(
    p_init=p, kappa=kap, theta=thet, sigma=sig, size=100_000)

# Setting up a trading environment from RL agents
