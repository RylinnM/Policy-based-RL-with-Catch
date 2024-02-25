#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custamizable Catch Environment
By Thomas Moerland
Leiden University, The Netherlands
2022

Extended from Catch environment in Behavioural Suite: https://arxiv.org/pdf/1908.03568.pdf
"""
import sys
import matplotlib
matplotlib.use('TkAgg') #'Qt5Agg') # 'TkAgg'
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from scipy.signal import savgol_filter

ACTION_EFFECTS = (-1, 0, 1) # left, idle right. 
OBSERVATION_TYPES = ['pixel','vector']

class Catch():
    """
    Reinforcement learning environment where we need to move a paddle to catch balls that drop from the top of the screen. 
    
    -----------
    |     o   |
    |         |
    |         | 
    |         |
    |   _     |
    -----------
    
    o = ball
    _ = paddle
    
    
    State space: 
        - The width and height of the problem can be adjusted with the 'rows' and 'columns' argument upon initialization.
        - The observation space can either be a vector with xy-locations of paddle and lowest ball, 
        or a binary two-channel pixel array, with the paddle location in the first channel, and all balls in the second channel.
        This can be determined with the 'observation_type' argument upon initialization. 
        
    Action space: 
        - Each timestep the paddle can move left, right or stay idle.
        
    Reward function: 
        - When we catch a ball when it reaches the bottom row, we get a reward of +1. 
        - When we miss a ball that reaches the bottom row, we get a penalty of -1. 
        - All other situaties have a reward of 0.  
        
    Dynamcics function: 
        - Balls randomly drop from one of the possible positions on top of the screen.
        - The speed of dropping can be adjusted with the 'speed' parameter. 
        
    Termination: 
        - The task terminates when 1) we reach 'max_steps' total steps (to be set upon initialization), 
        or 2) we miss 'max_misses' total balls (to be set upon initialization). 
    
    """
  
    def __init__(self, rows: int = 7, columns: int = 7, speed: float = 1.0, 
                 max_steps: int = 250, max_misses: int = 10, 
                 observation_type: str = 'pixel', seed = None, 
                 ):
        """ Arguments: 
        rows: the number of rows in the environment grid.
        columns: number of columns in the environment grid.
        speed: speed of dropping new balls. At 1.0 (default), we drop a new ball whenever the last one drops from the bottom. 
        max_steps: number of steps after which the environment terminates.
        max_misses: number of missed balls after which the environment terminates (when this happens before 'max_steps' is reached).
        observation_type: type of observation, either 'vector' or 'pixel'. 
              - 'vector': observation is a vector of length 3:  [x_paddle,x_lowest_ball,y_lowest_ball]
              - 'pixel': observation is an array of size [rows x columns x 2], with one hot indicator for the paddle location in the first channel,
              and one-hot indicator for every present ball in the second channel. 
        seed: environment seed. 
        """
        if observation_type not in OBSERVATION_TYPES:
            raise ValueError('Invalid "observation_type". Needs to be in  {}'.format(OBSERVATION_TYPES))
        if speed <= 0.0:
            raise ValueError('Dropping "speed" should be larger than 0.0')
        
        # store arguments
        self._rng = np.random.RandomState(seed)
        self.rows = rows
        self.columns = columns
        self.speed = speed
        self.max_steps = max_steps
        self.max_misses = max_misses
        self.observation_type = observation_type
        
        # compute the drop interval 
        self.drop_interval = max(1,rows // speed) # compute the interval towards the next drop, can never drop below 1
        if speed != 1.0 and observation_type == 'vector': 
            print('Warning: You use speed > 1.0, which means there may be multiple balls in the screen at the same time.' +
                  'However, with observation_type = vector, only the xy location of *lowest* ball is visible to the agent' +
                  ' (to ensure a fixed length observation vector')
            
        # Initialize counter
        self.total_timesteps = None 
        self.fig = None
        self.action_space = spaces.Discrete(3,)
        if self.observation_type == 'vector':
            self.observation_space = spaces.Box(low=np.array((0,0,0)), high=np.array((self.columns, self.columns, self.rows)), dtype=int)
        elif self.observation_type == 'pixel':
            self.observation_space = spaces.Box(low=np.zeros((self.rows, self.columns, 2)), high=np.ones((self.rows, self.columns, 2)), dtype=int)

    def reset(self):
        ''' Reset the problem to empty board with paddle in the middle bottom and a first ball on a random location in the top row '''
        # reset all counters
        self.total_timesteps = 0
        self.total_reward = 0
        self.r = '-'
        self.missed_balls = 0
        self.time_till_next_drop = self.drop_interval
        self.terminal = False

        # initialized problem
        self.paddle_xy = [self.columns // 2, 0] # paddle in the bottom middle
        self.balls_xy = [] # empty the current balls
        self._drop_new_ball() # add the first ball
        s0 = self._get_state() # get first state
        return s0

    def step(self,a):
        ''' Forward the environment one step based on provided action a ''' 
        
        # Check whether step is even possible
        if self.total_timesteps is None: 
            ValueError("You need to reset() the environment before you can call step()")
        elif self.terminal: 
            ValueError("Environment has terminated, you need to call reset() first")
        
        # Move the paddle based on the chosen action
        self.paddle_xy[0] = np.clip(self.paddle_xy[0] + ACTION_EFFECTS[a],0,self.columns -1) 
        
        # Drop all balls one step down
        for ball in self.balls_xy: 
            ball[1] -= 1

        # Check whether lowest ball dropped from the bottom
        if len(self.balls_xy) > 0:         # there is a ball present
            if self.balls_xy[0][1] < 0:    # the lowest ball reached below the bottom
                del self.balls_xy[0]
        
        # Check whether we need to drop a new ball
        self.time_till_next_drop -= 1
        if self.time_till_next_drop == 0:
            self._drop_new_ball()
            self.time_till_next_drop = self.drop_interval 
            
        # Compute rewards
        if (len(self.balls_xy) == 0) or (self.balls_xy[0][1] != 0): # no ball present at bottom row
            r = 0.0 
        elif self.balls_xy[0][0] == self.paddle_xy[0]: # ball and paddle location match, caught a ball
            r = 1.0 
        else: # missed the ball
            r = -1.0
            self.missed_balls += 1
        
        # Compute termination
        self.total_timesteps += 1
        if (self.total_timesteps == self.max_steps) or (self.missed_balls == self.max_misses):
            self.terminal = True
        else:
            self.terminal = False
        
        self.r = r
        self.total_reward += r
        return self._get_state(), r, self.terminal
    
    def render(self,step_pause=0.3):
        ''' Render the current environment situation '''
        if self.total_timesteps is None: 
            ValueError("You need to reset() the environment before you render it")
        
        # In first call initialize figure
        if self.fig == None:
            self._initialize_plot()

        # Set all colors to white
        for x in range(self.columns):
            for y in range(self.rows):
                if self.paddle_xy == [x,y]: # hit the agent location
                    if [x,y] in self.balls_xy: # agent caught a ball
                        self.patches[x][y].set_color('g')
                    else: 
                        self.patches[x][y].set_color('y')
                elif [x,y] in self.balls_xy: # hit a ball location without agent
                    if y == 0: # missed the ball
                        self.patches[x][y].set_color('r')
                    else: # just a ball
                        self.patches[x][y].set_color('w')
                else: # empty spot
                    self.patches[x][y].set_color('k')
        #plt.axis('off')
        
        self.label.set_text('Reward:  {:<5}            Total reward:  {:<5}     \nTotal misses: {:>2}/{:<2}     Timestep: {:>3}/{:<3}'.format(
            self.r,self.total_reward,self.missed_balls,self.max_misses,self.total_timesteps,self.max_steps))

        # Draw figure
        plt.pause(step_pause)                  

        
    def _initialize_plot(self):
        ''' initializes the catch environment figure ''' 
        self.fig,self.ax = plt.subplots()
        self.fig.set_figheight(self.rows)
        self.fig.set_figwidth(self.columns)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim([0,self.columns])
        self.ax.set_ylim([0,self.rows]) 
        self.ax.axes.xaxis.set_visible(False)
        self.ax.axes.yaxis.set_visible(False)

        self.patches = [ [[] for x in range(self.rows)] for y in range(self.columns)]
        for x in range(self.columns):
            for y in range(self.rows):
                self.patches[x][y] = Rectangle((x, y),1,1, linewidth=0.0, color = 'k') 
                self.ax.add_patch(self.patches[x][y]) 
                
        self.label = self.ax.text(0.01,self.rows + 0.2,'', fontsize=20, c='k')
        
    def _drop_new_ball(self):
        ''' drops a new ball from the top ''' 
        self.balls_xy.append([self._rng.randint(self.columns),self.rows-1])#0])
            
    def _get_state(self):
        ''' Returns the current agent observation '''
        if self.observation_type == 'vector': 
            if len(self.balls_xy) > 0: # balls present
                s = np.append(self.paddle_xy[0],self.balls_xy[0]).astype('float32') # paddle xy and ball xy 
            else: 
                s = np.append(self.paddle_xy[0],[-1,-1]).astype('float32') # no balls, impute (-1,-1) in state for no ball present
        elif self.observation_type == 'pixel':
            s = np.zeros((self.columns,self.rows, 2), dtype=np.float32)
            s[self.paddle_xy[0], self.paddle_xy[1], 0] = 1.0 # set paddle indicator in first slice
            for ball in self.balls_xy:
                s[ball[0], ball[1], 1] = 1.0 # set ball indicator(s) in second slice
        else:
            raise ValueError('observation_type not recognized, needs to be in {}'.format(OBSERVATION_TYPES))
        return s

class Hyperparameters():
    def __init__(self, entro_param = 0.01, gamma = 0.99, lr = 0.001, hidden_size = 64, size = 7, speed = 1.0, observation_type = 'vector', rows = 7, columns = 7, num_episodes = 500):
        #Model Parameters
        self.entro_param = entro_param
        self.gamma = gamma
        self.lr = lr
        self.hidden_size = hidden_size

        #Environment parameters
        self.rows = rows
        self.columns = columns
        self.speed = speed
        self.max_steps = 250
        self.max_misses = 10
        self.observation_type = observation_type  # 'vector'
        self.seed = None
        self.num_episodes = num_episodes


class Actor(nn.Module):
    def __init__(self, hyp):
        super(Actor, self).__init__()
        if hyp.observation_type == 'pixel':
            state_size = hyp.rows * hyp.columns * 2
        else:
            state_size = 1 * 3
        action_size = 3
        hidden_size = hyp.hidden_size
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

class Critic(nn.Module):
    def __init__(self, hyp):
        super(Critic, self).__init__()
        if hyp.observation_type == 'pixel':
            state_size = hyp.rows * hyp.columns * 2
        else:
            state_size = 1 * 3

        hidden_size = hyp.hidden_size
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Policy(nn.Module):
    def __init__(self, hyp):
        super(Policy, self).__init__()
        if hyp.observation_type == 'pixel':
            state_size = hyp.rows * hyp.columns * 2
        else:
            state_size = 1 * 3

        action_size = 3
        hidden_size = hyp.hidden_size
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(self.fc3(x), dim=-1)
        return x



def update_policy(hyp, rewards,saved_log_probs, saved_entropys, optimizer, gamma):
    R = 0
    policy_loss = []
    returns = deque()
    returns.append(0)

    # calculate sampled trajectory rewards backwards
    for r in reversed(rewards):
        R = returns[0]
        returns.appendleft(gamma * R + r)

    # calculate the overall loss of one trajectory
    returns = torch.Tensor(returns)
    returns = returns - returns.mean()
    for log_prob, entropy , R in zip(saved_log_probs,saved_entropys, returns):
        policy_loss.append(-log_prob * R - hyp.entro_param * entropy)
    policy_loss = torch.cat(policy_loss).sum()

    # update gradient and policy
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    return policy_loss.detach()

def select_action(state, policy):
    # input states
    state = torch.from_numpy(state).float().unsqueeze(0)
    # predict the probability of the possible actions
    probs = policy(state)
    # entropy regularization??
    dist = Categorical(probs)
    # sample the action
    action = dist.sample()
    return dist, action

def REINFORCE(hyp):
    env = Catch(rows=hyp.rows, columns=hyp.columns, speed=hyp.speed, max_steps=hyp.max_steps,
                max_misses=hyp.max_misses, observation_type=hyp.observation_type, seed=hyp.seed)
    env.reset()
    # define policy and optimizer

    policy = Policy(hyp)
    optimizer = optim.Adam(policy.parameters(), lr=hyp.lr) # Both work
    #optimizer = optim.RMSprop(policy.parameters(), lr=0.01) # Both work


    scores = []
    avg_scores = []
    policy_losses = []

    num_episodes = hyp.num_episodes
    gamma = hyp.gamma
    max_steps = hyp.max_steps

    for i_episode in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        saved_log_probs = []
        saved_entropys = []
        rewards = []
        for t in range(max_steps):
            dist, action = select_action(state, policy)
            saved_log_probs.append(dist.log_prob(action))
            saved_entropys.append(dist.entropy().mean())

            action = action.item()
            state, reward, done = env.step(action)
            rewards.append(reward)
            ep_reward += reward
            if done:
                break
        scores.append(ep_reward)
        avg_scores.append(np.mean(scores[-50:]))
        # update after every trajectory trace

        if i_episode <= 500:
            policy_losses.append(update_policy(hyp, rewards, saved_log_probs, saved_entropys, optimizer, gamma))

        if i_episode % 50 == 0:
            print(avg_scores[-1])
            print('Episode {}\tEpisode Score: {:.2f}'.format(i_episode, np.mean(scores[i_episode:i_episode+100], dtype=np.float32)))

    return (scores, avg_scores, policy_losses)

def actor_critic_baseline_subtraction_bootstrapping(hyp, do_bootstrapping = True, do_baseline = True):

    
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device is potentially faster")
    actor_model = Critic(hyp).to(device)
    critic_model = Critic(hyp).to(device)
    print("actor model: ", actor_model)
    print("critic model: ", critic_model)

    # Initialize environment and Q-array
    env = Catch(rows=hyp.rows, columns=hyp.columns, speed=hyp.speed, max_steps=hyp.max_steps,
                max_misses=hyp.max_misses, observation_type=hyp.observation_type, seed=hyp.seed)

    s = env.reset()
    step_pause = 0.3  # the pause between each plot
    #env.render(step_pause)

    actor = Actor(hyp)
    critic = Critic(hyp)
    optimizer_actor = optim.Adam(actor.parameters(), lr=hyp.lr)
    optimizer_critic = optim.Adam(critic.parameters(), lr=hyp.lr)

    scores = []
    avg_scores = []
    actor_losses = []
    critic_losses = []
    num_episodes = hyp.num_episodes
    gamma = hyp.gamma

    baseline = 0
    num_baseline_updates = 0

    for i_episode in range(num_episodes):

        state = env.reset()
        done = False
        score = 0
        while not done:

            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action_probs = actor(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            next_state, reward, done = env.step(action.item())

            score += reward
            value = critic(state_tensor)

            if do_bootstrapping:
                next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)
                next_value = critic(next_state_tensor)


            if do_baseline and do_bootstrapping:
                # Update critic with baseline subtraction and bootstrapping
                baseline = critic(state_tensor).detach()
                target = reward + gamma * next_value
                td_error = target - baseline
                loss_critic = F.smooth_l1_loss(value, target - baseline)

            elif do_bootstrapping:
                td_error = reward + gamma * next_value
                loss_critic = F.smooth_l1_loss(value, td_error)

            elif do_baseline:
                td_error = reward - torch.tensor(baseline)
                loss_critic = F.smooth_l1_loss(value, td_error)
            
            optimizer_critic.zero_grad()
            loss_critic.backward()
            optimizer_critic.step()

            # Update actor
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()

            if do_baseline:
                advantage = td_error.detach()
                loss_actor = -log_prob * advantage - hyp.entro_param * entropy
            else:
                loss_actor = -log_prob * td_error.item() - hyp.entro_param * entropy

            if i_episode <= 500:
                optimizer_actor.zero_grad()
                loss_actor.backward()
                optimizer_actor.step()

            if done:
                state = env.reset()
                scores.append(score)
                avg_scores.append(np.mean(scores[-50:]))
                actor_losses.append(loss_actor.item())
                critic_losses.append(loss_critic.item())

                if do_baseline: 
                    if do_bootstrapping == False:
                        num_baseline_updates += 1
                    baseline += (value - baseline) / num_baseline_updates

            else:
                state = next_state
        
        print('Episode {}\tEpisode Score: {:.2f}'.format(i_episode, scores[i_episode]))

    #plot_training_progress(scores, avg_scores, actor_losses, critic_losses, do_bootstrapping, do_baseline)

    return(scores, avg_scores, actor_losses, critic_losses)

class Tool:
    # Computes softmax
    def softmax(x, temp):
        # scale by temperature
        x = x / temp
        # subtract max to prevent overflow of softmax
        z = x - np.max(x)
        # compute softmax
        return np.exp(z) / np.sum(np.exp(z))

    def smooth(y, window, poly=2):
        '''
        y: vector to be smoothed
        window: size of the smoothing window '''
        return savgol_filter(y, window, poly)

class HyperTuning:
    # HT class, used for hyperparameter tuning
    def __init__(self):
        self.entro_params = [0.01, 0.001, 0.02]
        self.gammas = [0.99, 0.999, 0.9]
        self.lrs = [0.001, 0.0002, 0.005]
        self.lrs2 = [0.1, 0.01, 0.2]
        self.hidden_sizes = [64, 32, 128]

        self.sizes = [5, 7, 12]
        self.speeds = [0.5, 1, 2]
        self.observation_types = ['pixel', 'vector']
        self.variants = [[7, 14, 0.5, 'pixel'], [14, 7, 3, 'pixel'], [7, 14, 0.5, 'vector']] 



def OptimalEnvironmentVariation(command):
    """Plot the training progress of the actor-critic algorithm."""
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    n_repetitions = 3
    hypertunes = HyperTuning()
    if command == "size":

        for size in hypertunes.sizes: 

            hyp = Hyperparameters(entro_param = 0.01, gamma = 0.9, lr = 0.01, hidden_size = 128, rows = size, columns = size)
            scores_results = []
            avg_policy_losses = []

            for rep in range(n_repetitions): # Loop over repetitions
                print("repetition ", rep)
                scores, avg_scores, policy_losses = REINFORCE(hyp)
                scores_results.append( scores )
                avg_policy_losses.append( policy_losses )

            scores = np.mean(scores_results,axis=0)
            policy_losses = np.mean(avg_policy_losses,axis=0)

            policy_losses = Tool.smooth(policy_losses, 21)
            scores = Tool.smooth(scores, 21)

            # Plot the raw scores
            axs[0].plot(scores, label=f"size: {size}")
    
            # Plot the rolling average of the scores
            axs[1].plot(policy_losses, label=f"size: {size}")
    
        # Add axis labels and titles
        axs[0].set_title('Average scores')
        axs[1].set_title('Policy_losses')

        for ax in axs.flat:
            ax.set(xlabel='Episode', ylabel='Value')
        fig.suptitle('size with optimal REINFORCE', fontsize=16)

        fontsize = 12
        for ax in axs.flat:
            ax.tick_params(labelsize=fontsize)
            ax.set_title(ax.get_title(), fontsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
            ax.legend()
        fig.tight_layout()

        # Show the plot
        plt.savefig('size with optimal REINFORCE')

    if command == "speed":

        for speed in hypertunes.speeds: 

            hyp = Hyperparameters(entro_param = 0.01, gamma = 0.9, lr = 0.01, hidden_size = 128, speed = speed)
            scores_results = []
            avg_policy_losses = []

            for rep in range(n_repetitions): # Loop over repetitions
                print("repetition ", rep)
                scores, avg_scores, policy_losses = REINFORCE(hyp)
                scores_results.append( scores )
                avg_policy_losses.append( policy_losses )

            scores = np.mean(scores_results,axis=0)
            policy_losses = np.mean(avg_policy_losses,axis=0)

            policy_losses = Tool.smooth(policy_losses, 21)
            scores = Tool.smooth(scores, 21)

            # Plot the raw scores
            axs[0].plot(scores, label=f"speed: {speed}")
    
            # Plot the rolling average of the scores
            axs[1].plot(policy_losses, label=f"speed: {speed}")
    
        # Add axis labels and titles
        axs[0].set_title('Average scores')
        axs[1].set_title('Policy_losses')

        for ax in axs.flat:
            ax.set(xlabel='Episode', ylabel='Value')
        fig.suptitle('speed with optimal REINFORCE', fontsize=16)

        fontsize = 12
        for ax in axs.flat:
            ax.tick_params(labelsize=fontsize)
            ax.set_title(ax.get_title(), fontsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
            ax.legend()
        fig.tight_layout()

        # Show the plot
        plt.savefig('speed with optimal REINFORCE')


    if command == "observation_type":

        for observation_type in hypertunes.observation_types: 

            hyp = Hyperparameters(entro_param = 0.01, gamma = 0.9, lr = 0.01, hidden_size = 128, observation_type = observation_type)
            scores_results = []
            avg_policy_losses = []

            for rep in range(n_repetitions): # Loop over repetitions
                print("repetition ", rep)
                scores, avg_scores, policy_losses = REINFORCE(hyp)
                scores_results.append( scores )
                avg_policy_losses.append( policy_losses )

            scores = np.mean(scores_results,axis=0)
            policy_losses = np.mean(avg_policy_losses,axis=0)

            policy_losses = Tool.smooth(policy_losses, 21)
            scores = Tool.smooth(scores, 21)

            # Plot the raw scores
            axs[0].plot(scores, label=f"observation_type: {observation_type}")
    
            # Plot the rolling average of the scores
            axs[1].plot(policy_losses, label=f"observation_type: {observation_type}")
    
        # Add axis labels and titles
        axs[0].set_title('Average scores')
        axs[1].set_title('Policy_losses')

        for ax in axs.flat:
            ax.set(xlabel='Episode', ylabel='Value')
        fig.suptitle('observation_type with optimal REINFORCE', fontsize=16)

        fontsize = 12
        for ax in axs.flat:
            ax.tick_params(labelsize=fontsize)
            ax.set_title(ax.get_title(), fontsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
            ax.legend()
        fig.tight_layout()

        # Show the plot
        plt.savefig('observation_type with optimal REINFORCE')


    if command == "variant":

        for variant in hypertunes.variants: 

            hyp = Hyperparameters(entro_param = 0.01, gamma = 0.9, lr = 0.01, hidden_size = 128, rows = variant[0], columns = variant[1], speed = variant[2], observation_type = variant[3])
            scores_results = []
            avg_policy_losses = []

            for rep in range(n_repetitions): # Loop over repetitions
                print("repetition ", rep)
                scores, avg_scores, policy_losses = REINFORCE(hyp)
                scores_results.append( scores )
                avg_policy_losses.append( policy_losses )

            scores = np.mean(scores_results,axis=0)
            policy_losses = np.mean(avg_policy_losses,axis=0)

            policy_losses = Tool.smooth(policy_losses, 21)
            scores = Tool.smooth(scores, 21)

            # Plot the raw scores
            axs[0].plot(scores, label=f"rows:{variant[0]}, colums:{variant[1]}, speed:{variant[2]} ,type:{variant[3]}")
    
            # Plot the rolling average of the scores
            axs[1].plot(policy_losses, label=f"rows:{variant[0]}, colums:{variant[1]}, speed:{variant[2]} ,type:{variant[3]}")
    
        # Add axis labels and titles
        axs[0].set_title('Average scores')
        axs[1].set_title('Policy_losses')

        for ax in axs.flat:
            ax.set(xlabel='Episode', ylabel='Value')
        fig.suptitle('variant with optimal REINFORCE', fontsize=16)

        fontsize = 12
        for ax in axs.flat:
            ax.tick_params(labelsize=fontsize)
            ax.set_title(ax.get_title(), fontsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
            ax.legend()
        fig.tight_layout()

        # Show the plot
        plt.savefig('variant with optimal REINFORCE')
  


def plot_training_progress(do_bootstrapping, do_baseline, command):
    """Plot the training progress of the actor-critic algorithm."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    n_repetitions = 5
    hypertunes = HyperTuning()
    
    bootraptext = "bootstrapping"
    baselinetext = "baseline subtraction"
    text = ""

    if do_bootstrapping: 
        text = bootraptext
        if do_baseline:
            text += " and "
            text += baselinetext
    else: 
        text = baselinetext

    if command == "entro_param":

        for entro_param in hypertunes.entro_params: 

            hyp = Hyperparameters(entro_param = entro_param, gamma = 0.99, lr = 0.001, hidden_size = 64)
            scores, avg_scores, actor_losses, critic_losses = actor_critic_baseline_subtraction_bootstrapping(hyp, do_bootstrapping, do_baseline)

            scores_results = []
            avg_scores_results = []
            actor_losses_results = []
            critic_losses_results = []

            for rep in range(n_repetitions): # Loop over repetitions
                print("repetition ", rep)
                scores, avg_scores, actor_losses, critic_losses = actor_critic_baseline_subtraction_bootstrapping(hyp, do_bootstrapping, do_baseline)
                scores_results.append( scores )
                avg_scores_results.append( avg_scores )
                actor_losses_results.append( actor_losses)
                critic_losses_results.append(critic_losses)

            scores = np.mean(scores_results,axis=0)
            avg_scores = np.mean(avg_scores_results,axis=0)
            actor_losses = np.mean(actor_losses_results,axis=0)
            critic_losses = np.mean(critic_losses_results,axis=0)

            scores = Tool.smooth(scores, 31)
            actor_losses = Tool.smooth(actor_losses, 31)
            critic_losses = Tool.smooth(critic_losses, 31)

            # Plot the raw scores
            axs[0, 0].plot(scores, label=f"entro_param: {entro_param}")
    
            # Plot the rolling average of the scores
            axs[0, 1].plot(avg_scores, label=f"entro_param: {entro_param}")
    
            # Plot the actor loss over time
            axs[1, 0].plot(actor_losses, label=f"entro_param: {entro_param}")
    
            # Plot the critic loss over time
            axs[1, 1].plot(critic_losses, label=f"entro_param: {entro_param}")
    

        # Add axis labels and titles
        axs[0, 0].set_title('Raw scores')
        axs[0, 1].set_title('Rolling average scores')
        axs[1, 1].set_title('Critic loss')
        axs[1, 0].set_title('Actor loss')
        for ax in axs.flat:
            ax.set(xlabel='Episode', ylabel='Value')
        fig.suptitle('entro_param of Actor-Critic with ' + text, fontsize=16)

        fontsize = 12
        for ax in axs.flat:
            ax.tick_params(labelsize=fontsize)
            ax.set_title(ax.get_title(), fontsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
            ax.legend()
        fig.tight_layout()

        # Show the plot
        plt.savefig('entro_param of Actor-Critic with ' + text)

    if command == "gamma":

        for gamma in hypertunes.gammas: 

            hyp = Hyperparameters(entro_param = 0.001, gamma = gamma, lr = 0.001, hidden_size = 64)
            scores, avg_scores, actor_losses, critic_losses = actor_critic_baseline_subtraction_bootstrapping(hyp, do_bootstrapping, do_baseline)

            scores_results = []
            avg_scores_results = []
            actor_losses_results = []
            critic_losses_results = []

            for rep in range(n_repetitions): # Loop over repetitions
                print("repetition ", rep)
                scores, avg_scores, actor_losses, critic_losses = actor_critic_baseline_subtraction_bootstrapping(hyp, do_bootstrapping, do_baseline)
                scores_results.append( scores )
                avg_scores_results.append( avg_scores )
                actor_losses_results.append( actor_losses)
                critic_losses_results.append(critic_losses)

            scores = np.mean(scores_results,axis=0)
            avg_scores = np.mean(avg_scores_results,axis=0)
            actor_losses = np.mean(actor_losses_results,axis=0)
            critic_losses = np.mean(critic_losses_results,axis=0)

            scores = Tool.smooth(scores, 31)
            actor_losses = Tool.smooth(actor_losses, 31)
            critic_losses = Tool.smooth(critic_losses, 31)

            # Plot the raw scores
            axs[0, 0].plot(scores, label=f"gamma: {gamma}")
    
            # Plot the rolling average of the scores
            axs[0, 1].plot(avg_scores, label=f"gamma: {gamma}")
    
            # Plot the actor loss over time
            axs[1, 0].plot(actor_losses, label=f"gamma: {gamma}")
    
            # Plot the critic loss over time
            axs[1, 1].plot(critic_losses, label=f"gamma: {gamma}")
    

        # Add axis labels and titles
        axs[0, 0].set_title('Raw scores')
        axs[0, 1].set_title('Rolling average scores')
        axs[1, 1].set_title('Critic loss')
        axs[1, 0].set_title('Actor loss')
        for ax in axs.flat:
            ax.set(xlabel='Episode', ylabel='Value')
        fig.suptitle('gamma Actor-Critic with ' + text, fontsize=16)

        fontsize = 12
        for ax in axs.flat:
            ax.tick_params(labelsize=fontsize)
            ax.set_title(ax.get_title(), fontsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
            ax.legend()
        fig.tight_layout()

        # Show the plot
        plt.savefig('gamma Actor-Critic with ' + text)
        

    if command == "lr":

    
        for lr in hypertunes.lrs: 

            hyp = Hyperparameters(entro_param = 0.001, gamma = 0.99, lr = lr, hidden_size = 64)
            scores, avg_scores, actor_losses, critic_losses = actor_critic_baseline_subtraction_bootstrapping(hyp, do_bootstrapping, do_baseline)

            scores_results = []
            avg_scores_results = []
            actor_losses_results = []
            critic_losses_results = []

            for rep in range(n_repetitions): # Loop over repetitions
                print("repetition ", rep)
                scores, avg_scores, actor_losses, critic_losses = actor_critic_baseline_subtraction_bootstrapping(hyp, do_bootstrapping, do_baseline)
                scores_results.append( scores )
                avg_scores_results.append( avg_scores )
                actor_losses_results.append( actor_losses)
                critic_losses_results.append(critic_losses)

            scores = np.mean(scores_results,axis=0)
            avg_scores = np.mean(avg_scores_results,axis=0)
            actor_losses = np.mean(actor_losses_results,axis=0)
            critic_losses = np.mean(critic_losses_results,axis=0)

            scores = Tool.smooth(scores, 31)
            actor_losses = Tool.smooth(actor_losses, 31)
            critic_losses = Tool.smooth(critic_losses, 31)

            # Plot the raw scores
            axs[0, 0].plot(scores, label=f"lr: {lr}")
    
            # Plot the rolling average of the scores
            axs[0, 1].plot(avg_scores, label=f"lr: {lr}")
    
            # Plot the actor loss over time
            axs[1, 0].plot(actor_losses, label=f"lr: {lr}")
    
            # Plot the critic loss over time
            axs[1, 1].plot(critic_losses, label=f"lr: {lr}")
    

        # Add axis labels and titles
        axs[0, 0].set_title('Raw scores')
        axs[0, 1].set_title('Rolling average scores')
        axs[1, 1].set_title('Critic loss')
        axs[1, 0].set_title('Actor loss')
        for ax in axs.flat:
            ax.set(xlabel='Episode', ylabel='Value')
        fig.suptitle('lr Actor-Critic with ' + text, fontsize=16)

        fontsize = 12
        for ax in axs.flat:
            ax.tick_params(labelsize=fontsize)
            ax.set_title(ax.get_title(), fontsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
            ax.legend()
        fig.tight_layout()

        # Show the plot
        plt.savefig('lr Actor-Critic with ' + text)

    if command == "hidden_size":

        for hidden_size in hypertunes.hidden_sizes: 

            hyp = Hyperparameters(entro_param = 0.001, gamma = 0.99, lr = 0.001, hidden_size = hidden_size)
            scores, avg_scores, actor_losses, critic_losses = actor_critic_baseline_subtraction_bootstrapping(hyp, do_bootstrapping, do_baseline)

            scores_results = []
            avg_scores_results = []
            actor_losses_results = []
            critic_losses_results = []

            for rep in range(n_repetitions): # Loop over repetitions
                print("repetition ", rep)
                scores, avg_scores, actor_losses, critic_losses = actor_critic_baseline_subtraction_bootstrapping(hyp, do_bootstrapping, do_baseline)
                scores_results.append( scores )
                avg_scores_results.append( avg_scores )
                actor_losses_results.append( actor_losses)
                critic_losses_results.append(critic_losses)

            scores = np.mean(scores_results,axis=0)
            avg_scores = np.mean(avg_scores_results,axis=0)
            actor_losses = np.mean(actor_losses_results,axis=0)
            critic_losses = np.mean(critic_losses_results,axis=0)

            scores = Tool.smooth(scores, 31)
            actor_losses = Tool.smooth(actor_losses, 31)
            critic_losses = Tool.smooth(critic_losses, 31)

            # Plot the raw scores
            axs[0, 0].plot(scores, label=f"hidden_size: {hidden_size}")
    
            # Plot the rolling average of the scores
            axs[0, 1].plot(avg_scores, label=f"hidden_size: {hidden_size}")
    
            # Plot the actor loss over time
            axs[1, 0].plot(actor_losses, label=f"hidden_size: {hidden_size}")
    
            # Plot the critic loss over time
            axs[1, 1].plot(critic_losses, label=f"hidden_size: {hidden_size}")
    

        # Add axis labels and titles
        axs[0, 0].set_title('Raw scores')
        axs[0, 1].set_title('Rolling average scores')
        axs[1, 1].set_title('Critic loss')
        axs[1, 0].set_title('Actor loss')
        for ax in axs.flat:
            ax.set(xlabel='Episode', ylabel='Value')
        fig.suptitle('hidden_size Actor-Critic with ' + text, fontsize=16)

        fontsize = 12
        for ax in axs.flat:
            ax.tick_params(labelsize=fontsize)
            ax.set_title(ax.get_title(), fontsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
            ax.legend()
        fig.tight_layout()

        # Show the plot
        plt.savefig('hidden_size Actor-Critic with ' + text)

def plot_Optimal():


    n_repetitions = 3


    hyp = Hyperparameters(entro_param = 0.01, gamma = 0.9, lr = 0.01, hidden_size = 128, num_episodes = 1000)
    scores_results = []
    for rep in range(n_repetitions): # Loop over repetitions

        print("repetition ", rep)
        scores, avg_scores, policy_losses = REINFORCE(hyp)
        scores_results.append( scores )

    scores = np.mean(scores_results,axis=0)
    scores = Tool.smooth(scores, 21)
    plt.plot(scores, label=f"REINFORCE")


    hyp = Hyperparameters(entro_param = 0.001, gamma = 0.999, lr = 0.001, hidden_size = 64, num_episodes = 1000)
    scores_results = []
    for rep in range(n_repetitions): # Loop over repetitions

        print("repetition ", rep)
        scores, avg_scores, actor_losses, critic_losses = actor_critic_baseline_subtraction_bootstrapping(hyp, True, True)
        scores_results.append( scores )

    scores = np.mean(scores_results,axis=0)
    scores = Tool.smooth(scores, 21)
    plt.plot(scores, label=f"Actor_Critic Bootstrapping + Baseline")

    hyp = Hyperparameters(entro_param = 0.001, gamma = 0.9, lr = 0.001, hidden_size = 64, num_episodes = 1000)
    scores_results = []
    for rep in range(n_repetitions): # Loop over repetitions

        print("repetition ", rep)
        scores, avg_scores, actor_losses, critic_losses = actor_critic_baseline_subtraction_bootstrapping(hyp, True, False)
        scores_results.append( scores )

    scores = np.mean(scores_results,axis=0)
    scores = Tool.smooth(scores, 21)
    plt.plot(scores, label=f"Actor_Critic Bootstrapping")

    hyp = Hyperparameters(entro_param = 0.01, gamma = 0.999, lr = 0.0002, hidden_size = 64, num_episodes = 1000)
    scores_results = []
    for rep in range(n_repetitions): # Loop over repetitions

        print("repetition ", rep)
        scores, avg_scores, actor_losses, critic_losses = actor_critic_baseline_subtraction_bootstrapping(hyp, False, True)
        scores_results.append( scores )

    scores = np.mean(scores_results,axis=0)
    scores = Tool.smooth(scores, 21)
    plt.plot(scores, label=f"Actor_Critic Baseline")


    plt.title('Average scores')
    plt.xlabel='Episode'
    plt.ylabel='Value'
    plt.tick_params(labelsize= 12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Optimal hyperparameters')

def plot_training_REINFORCE(command):
    """Plot the training progress of the actor-critic algorithm."""
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    n_repetitions = 3
    hypertunes = HyperTuning()
    if command == "entro_param":

        for entro_param in hypertunes.entro_params: 

            hyp = Hyperparameters(entro_param = entro_param, gamma = 0.9, lr = 0.01, hidden_size = 128)
            scores_results = []
            avg_policy_losses = []

            for rep in range(n_repetitions): # Loop over repetitions
                print("repetition ", rep)
                scores, avg_scores, policy_losses = REINFORCE(hyp)
                scores_results.append( scores )
                avg_policy_losses.append( policy_losses )

            scores = np.mean(scores_results,axis=0)
            policy_losses = np.mean(avg_policy_losses,axis=0)

            policy_losses = Tool.smooth(policy_losses, 21)
            scores = Tool.smooth(scores, 21)

            # Plot the raw scores
            axs[0].plot(scores, label=f"entro_param: {entro_param}")
    
            # Plot the rolling average of the scores
            axs[1].plot(policy_losses, label=f"entro_param: {entro_param}")
    
        # Add axis labels and titles
        axs[0].set_title('Average scores')
        axs[1].set_title('Policy_losses')

        for ax in axs.flat:
            ax.set(xlabel='Episode', ylabel='Value')
        fig.suptitle('entro_param of REINFORCE', fontsize=16)

        fontsize = 12
        for ax in axs.flat:
            ax.tick_params(labelsize=fontsize)
            ax.set_title(ax.get_title(), fontsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
            ax.legend()
        fig.tight_layout()

        # Show the plot
        plt.savefig('entro_param of REINFORCE')

    if command == "gamma":

        for gamma in hypertunes.gammas: 

            hyp = Hyperparameters(entro_param = 0.001, gamma = gamma, lr = 0.01, hidden_size = 128)
            scores_results = []
            avg_policy_losses = []

            for rep in range(n_repetitions): # Loop over repetitions
                print("repetition ", rep)
                scores, avg_scores, policy_losses = REINFORCE(hyp)
                scores_results.append( scores )
                avg_policy_losses.append( policy_losses )

            scores = np.mean(scores_results,axis=0)
            policy_losses = np.mean(avg_policy_losses,axis=0)

            policy_losses = Tool.smooth(policy_losses, 21)
            scores = Tool.smooth(scores, 21)

            # Plot the raw scores
            axs[0].plot(scores, label=f"gamma: {gamma}")
    
            # Plot the rolling average of the scores
            axs[1].plot(policy_losses, label=f"gamma: {gamma}")
    
        # Add axis labels and titles
        axs[0].set_title('Average scores')
        axs[1].set_title('Policy_losses')

        for ax in axs.flat:
            ax.set(xlabel='Episode', ylabel='Value')
        fig.suptitle('gamma of REINFORCE', fontsize=16)

        fontsize = 12
        for ax in axs.flat:
            ax.tick_params(labelsize=fontsize)
            ax.set_title(ax.get_title(), fontsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
            ax.legend()
        fig.tight_layout()

        # Show the plot
        plt.savefig('gamma of REINFORCE')
        

    if command == "lr":

    
        for lr in hypertunes.lrs2: 

            hyp = Hyperparameters(entro_param = 0.001, gamma = 0.9, lr = lr, hidden_size = 128)
            scores_results = []
            avg_policy_losses = []

            for rep in range(n_repetitions): # Loop over repetitions
                print("repetition ", rep)
                scores, avg_scores, policy_losses = REINFORCE(hyp)
                scores_results.append( scores )
                avg_policy_losses.append( policy_losses )

            scores = np.mean(scores_results,axis=0)
            policy_losses = np.mean(avg_policy_losses,axis=0)

            policy_losses = Tool.smooth(policy_losses, 21)
            scores = Tool.smooth(scores, 21)

            # Plot the raw scores
            axs[0].plot(scores, label=f"lr: {lr}")
    
            # Plot the rolling average of the scores
            axs[1].plot(policy_losses, label=f"lr: {lr}")
    
        # Add axis labels and titles
        axs[0].set_title('Average scores')
        axs[1].set_title('Policy_losses')

        for ax in axs.flat:
            ax.set(xlabel='Episode', ylabel='Value')
        fig.suptitle('lr of REINFORCE', fontsize=16)

        fontsize = 12
        for ax in axs.flat:
            ax.tick_params(labelsize=fontsize)
            ax.set_title(ax.get_title(), fontsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
            ax.legend()
        fig.tight_layout()

        # Show the plot
        plt.savefig('lr of REINFORCE')

    if command == "hidden_size":

        for hidden_size in hypertunes.hidden_sizes: 

            hyp = Hyperparameters(entro_param = 0.001, gamma = 0.9, lr = 0.01, hidden_size = hidden_size)
            scores_results = []
            avg_policy_losses = []

            for rep in range(n_repetitions): # Loop over repetitions
                print("repetition ", rep)
                scores, avg_scores, policy_losses = REINFORCE(hyp)
                scores_results.append( scores )
                avg_policy_losses.append( policy_losses )

            scores = np.mean(scores_results,axis=0)
            policy_losses = np.mean(avg_policy_losses,axis=0)

            policy_losses = Tool.smooth(policy_losses, 21)
            scores = Tool.smooth(scores, 21)

            # Plot the raw scores
            axs[0].plot(scores, label=f"hidden_size: {hidden_size}")
    
            # Plot the rolling average of the scores
            axs[1].plot(policy_losses, label=f"hidden_size: {hidden_size}")
    
        # Add axis labels and titles
        axs[0].set_title('Average scores')
        axs[1].set_title('Policy_losses')

        for ax in axs.flat:
            ax.set(xlabel='Episode', ylabel='Value')
        fig.suptitle('hidden_size of REINFORCE', fontsize=16)

        fontsize = 12
        for ax in axs.flat:
            ax.tick_params(labelsize=fontsize)
            ax.set_title(ax.get_title(), fontsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
            ax.legend()
        fig.tight_layout()

        # Show the plot
        plt.savefig('hidden_size of REINFORCE')



if __name__ == '__main__':
    

    args = sys.argv[1:]
    if args[0] == '--tune':
        #try:
            for arg in args[1:]:
                
                 plot_training_progress(True, True, arg)
                 plot_training_progress(True, False, arg)
                 plot_training_progress(False, True, arg)
                 plot_training_REINFORCE(arg)

        #except: 
            #print("Invalid hyperparameter. Try: 'entro_param', 'gamma', 'lr', 'hidden_size'")

    if args[0] == '--optimal':
        plot_Optimal()

    if args[0] == '--env':
        #try:
            for arg in args[1:]:
                
                OptimalEnvironmentVariation(arg)

        #except: 
            #print("Invalid hyperparameter. Try: 'size', 'Speed', 'observation_type', 'variant'")       
