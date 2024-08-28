## Title
Developing a custom agent and custom environment for Reinforcement Learning in Python
for Trade Execution Analysis.

## Project Overview
This projet is aimed at developing a custom agent and a custom environment for
Reinforcement Learning projects. Thus, it has two main files, each one of them
containing the implementation of the agents and the environments separately.
For a more detailed description of the code itself, please go to Code structure
section in this README.md file.

## Installation and Setup
This project is composed of the two following files:

- agents.py : File containing the custom agents built in Tensorflow.
  All the code within this package is self contained, except for 
  a 'create_nn' function, which is imported from ML_AI/Algos/
  Miscellaneous/misc_utils.py file. In case you want to use
  agents.py, make sure to import that function as well.

- environments.py : File containing the custom environments built in
  Gymnasium, a fork of OpenAI's Gym Library, the reference library for
  Reinforcement Learning projects (https://gymnasium.farama.org/index.html)
  

### Packages
- Gymnasium 0.29.1
- Python 3.10.12
- Pandas 2.1.0
- Numpy 1.26.4
- Matplotlib 3.6.0
- Scipy 1.14.0
- Tensorflow 2.16.1

## Data Source
agents.py file on itself does not consume any input data. Nevertheless,
the environmnents need an external source of data in order to work. The
specific details on the input dataset are defined within each environment
class, concretely in the initialization part, and are saved as self.data
argument.

## Code structure
As mentioned above, this project is structured around two main packages:

- agents.py: This file contains custom implementations for DDQN agents
  in Tensorflow. In order to do so, it creates a ReplayBuffer object
  to store tuples of state, action, reward, state_after needed to train
  the agents. The core class of the package is DDQN class, which
  is an implementation of the original DDQN algorithm
  (link to the original paper https://www.arxiv.org/abs/1509.06461).
  The two main functions of DDQN class are:
    - Train: Given the settings of the class and a pre-defined environment,
    this function trains the agent to learn a specific policy to perform 
    a specific task: That is, given an observed state, s, choose the action
    that maximizes the expected return (i.e sum of future rewards).
    Thia function contains a lot of debugging information to asses the 
    quality of the learning process.
    - Choose_action: Given an input state, the agent makes use of usually
    a pre-learnt policy via train to decide which action to take. That is, it
    applies a given policy to a state s. Note this function can be used 
    without prior learning, albeit the action taken will not
    be an informed one.
  The other main class of the package is DDQN_TradeExecution, which is just
  a child class of DDQN with custom functions for training and choosing
  actions.
  
- environment.py: This file defines custom environemnts for specific
  Reinforcement Learning tasks. The main goal of the environments
  is to define what is the effect/reaction of the agent taking a specific
  action on it. At the moment of writting, August 2024, this class 
  only contains a single environment focused on trading execution 
  (in the future other environments will be created), TradeExecution 
  class. This class is a child class of Gym environment, which needs
  at least four different functions to be implemented by the user to
  be able to work correctly. These are:

    - init      =>  Initializes the 
    - reset     =>  This function resets the environment to its starting state
                    whenever the environment reaches a terminal state
                    or a state which is invalid. This function has an argument, id
                    , which sets the starting point of the dataset after resetting
                    it. By default the environment will NOT set the dataframe to its
                    first index, 0, otherwise we would train the agent over the same
                    data over and over, which is undesired.
    - render    =>  this function renders the data into a visual form. This is useful
                    for environments which involve some kind of robot control. In our
                    case, it will return None.
    - step      =>  core part of the environemt. This is where we implement what
                    happens when the agent takes an action on the environment. That is,
                    to which state s_prime we arrive after taking action a to the 
                    state s. The step function returns a tuple of four variables
                    (observation, reward, done, info), which are the state we arrive
                    after applying action a to state s, the reward the agent obtains
                    for taking such action, whether the arrived state is a terminal 
                    one and some debugging info respectively.

## License
Source code is licensed under the MIT license.

Contents of this site are © Copyright 2024 Axel Borasino Marques. All rights reserved.
