# Policy-based-RL-with-Catch

## Overview
This repository contains a reinforcement learning environment, `Customizable Catch Environment`, developed by Thomas Moerland at Leiden University, The Netherlands. The environment is an extension of the Catch environment from the Behavioural Suite, where the objective is to control a paddle to catch falling balls.

## Environment Description
- **State Space**: The size of the environment grid can be adjusted with 'rows' and 'columns' arguments. The observation space can be a vector of xy-locations of the paddle and the lowest ball or a two-channel pixel array with the paddle location in the first channel and all balls in the second.
- **Action Space**: The paddle can move left, right, or stay idle each timestep.
- **Reward Function**: 
  - +1 reward for catching a ball at the bottom row.
  - -1 penalty for missing a ball at the bottom row.
  - 0 reward in all other situations.
- **Dynamics Function**: Balls drop randomly from the top of the screen. The dropping speed can be adjusted.
- **Termination**: The task ends when a set number of total steps (`max_steps`) is reached or a certain number of total balls (`max_misses`) is missed.

### Initialization Parameters
- `rows`: Number of rows in the environment grid.
- `columns`: Number of columns in the environment grid.
- `speed`: Speed of dropping new balls.
- `max_steps`: Maximum steps after which the environment terminates.
- `max_misses`: Number of missed balls after which the environment terminates.

## Running the Code
To run the environment, use the following command-line arguments:
- `--tune`: For tuning hyperparameters like 'entro_param', 'gamma', 'lr', 'hidden_size'.
- `--optimal`: To plot the optimal training progress.
- `--env`: To execute optimal environment variation based on arguments.

Example:
```bash
python Main.py --tune gamma
```

## Dependencies
- Python 3.x
- numpy
- matplotlib
- gym
- torch
- other dependencies as required by the above
