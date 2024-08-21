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
specific details on the input dataset are defined within each environment,
concretely in the initialization part, and are saved on the self.data
argument.

## Code structure
As mentioned above, this project is structured around two main packages:

- agents.py: This file contains custom implementations for DDQN agents
  in Tensorflow. In order to do so, it creates a ReplayBuffer object
  to store tuples of state,action,reward,state_after needed to train
  the agents. The core class from which child classes are then
  derived is an implementation of DDQN algorithm 
  (link to the original paper https://www.arxiv.org/abs/1509.06461).
  The two main functions created in this class are:
    - Train: Given the settings of the class and a pre-defined environment,
    this function trains the agent to learn a specific policy to perform 
    a specific task. It contains a lot of debugging information to 
    asses the quality of the learning process.
    - Choose_action: Given an input state, the agent makes use of usually
    a pre-learnt policy via train to decide which action to take. It
    can be used without prior learning, albeit the action taken will not
    be an informed one.
  The rest of created classes are just childs of this one with custom
  functions for learning, choosing actions.

- environment.py: This file defines custom environemnts for specific
  Reinforcement Learning projects. At the moment of
  writting, August 2024, this class only contains a single environment
  focused on trading execution (in the future other environments will
  be created), TradeExecution class. 

  TO-DO: Finish explanation of main parts of environment.py class

## License
Source code is licensed under the MIT license.

Contents of this site are © Copyright 2024 Axel Borasino Marques. All rights reserved.
