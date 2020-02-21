
## TODO ##

# Build LSTM
# Build 1D CNN
# Eventually will need valid/test data, but that is easy


import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import csv
from itertools import islice

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'target', 'next_action'))

class Env(object):

    def __init__(self, assets, minutes_back, period, params):
        self.data_path = None
        self.assets = assets
        self.minutes_back = minutes_back
        self.train_period = period
        self.train_length = minutes_back*60/period
        self.params = params

        # Initial environment parameters
        self.asset_status = 0
        self.bought = 0
        self.hold_time = 0

        self.num_episodes = 1
        self.extra_values = 2  # Asset status, bought value, hold_time
        self.train_params = params  # ['ask']
        self.train_env = None
        self.train_ask = None

        # Normalization/conversion
        self.spread = None
        self.offset = None
        
    def load_data(self):
        try:
            self.train_env = torch.load('{}_ask_p{}s_l{}m_env.pt'.format(self.assets, self.train_period, self.minutes_back))
            self.train_ask = torch.load('{}_ask_p{}s_l{}m_ask.pt'.format(self.assets, self.train_period, self.minutes_back))
        except:
            self.data_path = '/Users/mattsonthieme/Documents/Academic.nosync/Projects/RLTrading/data/crypto/ETH/ETH_1s_4.csv'
            self.train_env, self.train_ask = self.parse_data(self.data_path, self.train_length, self.train_params, self.train_period)
            torch.save(self.train_env, '{}_ask_p{}s_l{}m_env.pt'.format(self.assets, self.train_period, self.minutes_back))
            torch.save(self.train_ask, '{}_ask_p{}s_l{}m_ask.pt'.format(self.assets, self.train_period, self.minutes_back))

    # Return the next state, adding necessary environmental parameters
    def update(self, state, asset_status, bought, hold_time):
        # Add the current asset status and bought value to the state to the state
        state = torch.cat((state, torch.tensor([asset_status]).type('torch.FloatTensor')), 0).to(device)
        state = torch.cat((state, torch.tensor([bought]).type('torch.FloatTensor')), 0).to(device)
        #state = torch.cat((state, torch.tensor([hold_time]).type('torch.FloatTensor')), 0).to(device)
        return state

    def normalize(self):
        # Normalize train_env/ask to [-1,1]
        self.spread = max(self.train_ask) - min(self.train_ask)
        self.offset = self.train_ask.mean().item()
        self.train_env = (self.train_env - self.offset)/(self.spread)
        self.train_ask = (self.train_ask - self.offset)/(self.spread)


    # Returns two values
    # 1) train_environment: a list train_length*#params long corresponding to a history of train_length at each period interval
    # 2) env_value: a list of floats corresponding to the ask value at the current timestep
    def parse_data(self, path, train_length, params, period):
        print("Processing data...")
        with open(self.data_path, 'r') as f:
            data = csv.reader(f)
            data = list(data)
            labels = data.pop(0)
            indices = [labels.index(i) for i in self.params]
            data = np.array(data)

            train_environment = []
            env_value = []

            # Multiplex periods of > 1 second
            for i in range(period):

                # Filter data s.t. only rows corresponding to that period remain
                period_indices = [j for j in range(i, len(data), period)]
                tempdata = data[period_indices]

                # Swap col/rows for easier access 
                tempdata = tempdata.transpose()

                # Slice indices to determine individual environment states
                begin = 0
                end = train_length
                
                while end < len(tempdata[0]):
                    train_elem = []
                    for j in indices:
                        train_elem.extend(tempdata[j][begin:end])
                    train_environment.append(train_elem)
                    env_value.append(tempdata[labels.index('ask')][end-1])

                    begin += 1
                    end += 1
                print("{}% complete...".format(int(100*float(i)/period)))

        train_environment = np.array(train_environment)
        train_environment = train_environment.astype(float)
        train_environment = torch.from_numpy(train_environment).type('torch.FloatTensor').to(device)
        env_value = np.array(env_value)
        env_value = env_value.astype(float)
        return train_environment, env_value


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # Save transitions to the memory buffer
    # Once full, add new transitions onto the front of the buffer
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity


    # Pull a random sample from memory
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def _len(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_dim, 500)
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, 100)
        self.fc4 = nn.Linear(100, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class Agent(object):

    def __init__(self, policy, state_size):

        # Model parameters
        self.input_dimension = state_size
        self.n_actions = 3  # Buy, hold, sell
        
        if policy == 'mlp':
            self.policy_net = DQN(self.input_dimension, self.n_actions).to(device)
            self.target_net = DQN(self.input_dimension, self.n_actions).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

        # State parameters
        self.state = None
        self.asset_status = 0
        self.bought = 0
        self.hold_time = 0
        self.gain_track = 1

        # Learning parameters
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.001
        self.EPS_DECAY = 5000
        self.TARGET_UPDATE = 80
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.total_steps = 0
        self.BATCH_SIZE = 128

        # Memory
        self.mem_capacity = 10000 
        self.memory = ReplayMemory(self.mem_capacity)

        # Financial parameters
        self.fee = 0.00075  # 0.075% for Binance
        self.investment = 1000
        self.losses = []
        self.gains = []
        self.episode_profit = 0
        self.initial_market_value = 0
        self.session_begin_value = 0

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def report(self, ask, spread, offset):
        if len(self.gains) >= 20:

            print(self.gains)
            print(' ')
            print(self.losses)
            print(' ')
            self.episode_profit += sum(self.gains) + sum(self.losses)
            print("Market moved ${} over the session".format(round((ask*spread + offset) - self.session_begin_value,2)))
            print("Start: ${}, current: ${}".format(round(self.initial_market_value*spread + offset,3), round(ask*spread + offset,3)))
            print("     Session wins: {} @ ${}".format(len(self.gains), sum(self.gains)/len(self.gains)))
            print("     Session loss: {} @ ${}".format(len(self.losses), sum(self.losses)/len(self.losses)))
            print("     Session Net:  ${}".format(sum(self.gains) + sum(self.losses)))  # Losses are already negative
            print("     Episode total: ${}\n\n\n".format(self.episode_profit))

            self.session_begin_value = ask*spread + offset
            self.gains = []
            self.losses = []

    # Return asset status, bought value, 
    def take_action(self, state, ask, next_env, next_ask):
        rand = random.random()
        epsilon_threshold = self.EPS_END + (self.EPS_START - self.EPS_END)*math.exp(-1. * self.total_steps/self.EPS_DECAY)
        self.total_steps += 1
        if rand > epsilon_threshold:
            with torch.no_grad():
                action = self.policy_net(state).max(0)[1].view(1,1)  # Returns the index of the maximum output in a 1x1 tensor
        else:
            #print("Chose randomly ({})...".format(epsilon_threshold))
            action = torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

        reward = self.reward_calc(action, self.asset_status, self.investment, self.bought, self.hold_time, self.fee, ask)

        # Calculate target values
        target = next_env
        # Add the current asset status and bought value to the next
        target = torch.cat((target, torch.tensor([self.asset_status]).type('torch.FloatTensor')), 0).to(device)
        target = torch.cat((target, torch.tensor([self.bought]).type('torch.FloatTensor')), 0).to(device)
        #target = torch.cat((target, torch.tensor([self.hold_time]).type('torch.FloatTensor')), 0).to(device)

        # Push memory into buffer
        next_action = 0
        next_reward = 0
        with torch.no_grad():
            sold_target = next_ask
            next_action = self.target_net(target).max(0)[1].view(1,1)

            # All local values were updated with the last reward_calc
            next_reward = self.target_reward_calc(next_action, self.asset_status, self.investment, self.bought, self.hold_time, self.fee, next_ask)

        self.memory.push(state, action, reward, target, next_reward)

    # This separate calculator exists so we don't update the agen't state with the target reward calculation. Not ideal...
    def target_reward_calc(self, action, asset_status, investment, bought, hold_time, fee, ask):

        reward = 0
        # Money is in our wallet
        if self.asset_status == 0:

            # Buying is legal
            if action == 0:
                reward = 0

            # Holding is legal
            if action == 1:
                reward = 0

            # Selling is illegal
            if action == 2:
                reward = -100

        # Money is in an asset
        if self.asset_status == 1:

            # Buying is illegal
            if action == 0:
                reward = -100

            # Holding is legal
            if action == 1:
                reward = 0

            # Selling is legal
            if action == 2:
                buy_cost = self.investment*self.fee
                bought_shares = (self.investment - buy_cost)/self.bought
                sold_revenue = bought_shares*ask
                sell_cost = sold_revenue*self.fee
                reward = sold_revenue - sell_cost - buy_cost - self.investment
        
        return torch.tensor([reward]).type('torch.FloatTensor')

    ### Try not penalizing it for making the wrong choice, just for making legal choices
    def reward_calc(self, action, asset_status, investment, bought, hold_time, fee, ask):

        # Money is in our wallet
        if self.asset_status == 0:

            # Buying is legal
            if action == 0:
                self.bought = ask
                self.asset_status = 1  # Money is now in the asset
                reward = 0

            # Holding is legal
            if action == 1:
                self.hold_time += 1
                reward = 0

            # Selling is illegal
            if action == 2:
                self.hold_time += 1
                reward = -100

        # Money is in an asset
        if self.asset_status == 1:

            # Buying is illegal
            if action == 0:
                self.hold_time += 1
                reward = -100

            # Holding is legal
            if action == 1:
                self.hold_time += 1
                reward = 0

            # Selling is legal
            if action == 2:
                self.asset_status = 0  # Money is back in our wallet
                self.hold_time = 0

                buy_cost = self.investment*self.fee
                bought_shares = (self.investment - buy_cost)/self.bought
                sold_revenue = bought_shares*ask
                sell_cost = sold_revenue*self.fee
                reward = sold_revenue - sell_cost - buy_cost - self.investment

                if reward > 0:
                    self.gain_track += 1
                    self.gains.append(reward)
                if reward <= 0:
                    self.losses.append(reward)

        return torch.tensor([reward]).type('torch.FloatTensor')

    def optimize_model(self, BATCH_SIZE):

        if self.memory._len() < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)

        batch_state = []
        batch_action = []
        batch_reward = []
        batch_target = []
        batch_next_action = []
        for i, trans in enumerate(transitions):
            batch_state.append(trans.state)
            batch_action.append(trans.action[0])
            batch_reward.append(trans.reward[0])
            batch_target.append(trans.target[0])
            batch_next_action.append(trans.next_action[0])

        batch_state = torch.stack(batch_state).to(device)
        batch_action = torch.stack(batch_action).to(device)
        batch_reward = torch.stack(batch_reward).to(device)
        batch_target = torch.stack(batch_target).to(device)
        batch_next_action = torch.stack(batch_next_action).to(device)

        # State-action values
        state_action_values = self.policy_net(batch_state).gather(1, batch_action)

        # Target values
        expected_state_action_values = (batch_next_action[0]*self.GAMMA) + batch_reward

        #loss = F.smooth_l1_loss(state_action_values, batch_reward.unsqueeze(1))
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()

    # Implement to actually send orders with ccxt
    def buy(self):
        raise NotImplementedError

    def sell(self):
        raise NotImplementedError

class execute(object):
    def __init__(self):

        # Params for initializing the environment
        self.asset = 'ETH'
        self.minutes_back = 10
        self.period = 15
        self.params = ['ask']

        # Initialize the environment
        self.env = Env(self.asset, self.minutes_back, self.period, self.params)
        self.env.load_data()
        self.env.normalize()
        self.extra = self.env.extra_values

        # Initialize the agent
        self.agent = Agent('mlp', self.env.train_env[0].size()[0]+self.extra)
        self.agent.initial_market_value = self.env.train_ask[0]

    def trade(self):
        # Iterate over states

        for episode in range(self.env.num_episodes):
            for i, state in enumerate(self.env.train_env):

                # Add env parameters to the state
                state = self.env.update(state, self.agent.asset_status, self.agent.bought, self.agent.hold_time)
                
                # Take an action
                self.agent.take_action(state, self.env.train_ask[i], self.env.train_env[i+1], self.env.train_ask[i+1])

                # Optimize the agent according to that action
                self.agent.optimize_model(self.agent.BATCH_SIZE)

                # Output training info
                self.agent.report(self.env.train_ask[i], self.env.spread, self.env.offset)

                # Update target network
                if self.agent.gain_track > self.agent.TARGET_UPDATE:
                    self.agent.gain_track = 0
                    print("Updating target net...")
                    self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
        
        # Save policy every episode
        torch.save(self.agent.policy_net.state_dict(),"saved_policy_{}_{}.pt".format('ETH',datetime.datetime.now()))


beast = execute()
beast.trade()

