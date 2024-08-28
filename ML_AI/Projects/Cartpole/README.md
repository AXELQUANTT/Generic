## Title
Using a custom implementation of DDQN algorithm to solve Cartpole-v0 environment.

## Project Overview
This projet is aimed at using an own developed Reinforcement Learning agent
known as Double Deep-Q Networks (DDQN) to solve a common motion control
environment known as Cartpole-v0. For a detailed explanation on how the
agent is built, please look into DDQN class in the following package: 
https://github.com/AXELQUANTT/Generic/blob/main/ML_AI/Algos/ReinforcementLearning/agents.py
This project has two main goals:
    - Debugging: DDQN class will be used for more complex environments, but this solvable
    environment Cartpole-v0 serves as a good proxy to check that indeed the implementation
    is correct.
    - Parameters effect: Among some of the parameters that affect the learning process to
    a greater extent we have:
                             - Network architecture: Number of layers of the neuronal networks
                               and number of neurons per layer. One of the parameters that was also tested was to use the action as input of the network, ACT_AS_IN which is just an implementation detail and had no major effect.
                             - Learning rate: Size of the step used in Adam optimizer.
                             - Type of target network update: We differentiate between
                             hard copy (i.e weights of policy network copied to the target network every x episodes) or soft copy (i.e weights of target policy are a weighted average of current policy network weights and the previous target network weights)
                             - Greedy step: Greedy factor is reduced by this amount on every step. This controls how big the exploration phase is.
    

## Installation and Setup
This project is composed of the two following files:

- analysis.ipynb : Ipython notebook containing the analysis on itself.
  You will see this package imports multiple functions from the utils.py
  package, but it also imports DDQN class and Agent_Performance 
  from ML_AI/Algos/ReinforcementLearning/agents.py and
  sns_lineplot from Library/plotting.py. Please import
  these libraries from AXEL_QUANTT/Generic repo to be able to run
  this notebook.

- utils.py : File containing auxiliary functions to be imported
  by analysis.ipynb.

### Packages
- Gymnasium 0.29.1
- Matplotlib 3.6.0
- Numpy 1.26.4
- Python 3.10.12
- Pandas 2.1.0
- Seaborn 0.13.2
- Tensorflow 2.16.1

## Data Source
The project's only source of data is embedded in Cartpole-v0 environment,
which is imported via gym.make("CartPole-v0"). For a more detailed
explanation about what this environment is about, please check
https://gymnasium.farama.org/environments/classic_control/cart_pole/

## Code structure
As mentioned above, the main package of this project is analysis.ipynb,
which is explained in detail below:

- analysis.ipynb: The Ipython notebook starts with defining all the
parameters that parametrize the DDQN agent. For a detailed
explanation of these, please go to 
https://github.com/AXELQUANTT/Generic/blob/main/ML_AI/Algos/ReinforcementLearning/agents.py
Once a set of default of settings are defined, some variations of them are defined and the learning process for each one of them is launched. The two main outputs 
of the learning process are the reward curves (i.e aggregate reward per episode) and the
losses curves (i.e aggregate loss per episode). The last part of the project computes
set of statistics on top of these curves to asses the quality of the learnt policy.
Since Reinforcement Learning algorithms exhibit a high degree of dependency on the
initial conditions, multiple trials with the same parameters are performed to obtain more statistically robust conclusions. One thing to note is that the learning process
is finished once all the episodes are concluded or when a default reward of 195
is obtained in the last 100 episodes of training.

## Results
The results are widely discussed in the markdown cells of the notebook, but just
as a brief summary we found:
    - Network architecture: From the three configurations studied, the most simple one
    , networks of 2 layers of 64 neurons per layer, seems to be complex enough to solve
    the environment.
    - Greedy policy: In general, bigger values of the greedy decay factor lead to more
    robust learning processes, at a cost of longer periods to actually solve the
    environment. The default value of 0.999 seems good enough though. 
    - Type of target update: The most robust training is obtained when a soft update
    policy is used to copy the weights of the policy network to the target update,
    with 99% of the weight assigned to the previous target network weights and 1%
    to the new policy weights.
    - Learning rate analysis: This parameter is the one affecting the overall
    learning experience to a higher degree. Out of three configurations used,
    0.005 seems to be the good enough one.

## License
Source code is licensed under the MIT license.

Contents of this site are © Copyright 2024 Axel Borasino Marques. All rights reserved.
