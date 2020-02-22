
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
                        ('state', 'properties', 'action', 'reward', 'target', 'next_action'))


# This model has three parts
# 1) An LSTM processing only price history data
# 2) A 'legality network' looking at asset_status, bought price, hold_time to determine if certain actions are legal
# 3) A 'decision network' looking at the output of 1) and 2) to make the final decision
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_layers, properties):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # Note: output_dim for LSTM = hidden_dim for now
        self.n_layers = n_layers
        self.lstm_layer = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)

        self.batch_size = 1
        self.seq_len = 1

        self.inp = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        self.hidden_state = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        self.cell_state = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        self.hidden = (self.hidden_state, self.cell_state)

        # Legality network
        prop_length = properties.size()[0]
        dec_size = 8  # Size of the final embedding, try some diff vals
        self.ln1 = nn.Linear(prop_length, 32)
        self.ln2 = nn.Linear(32, dec_size)

        # Decision network
        self.output_dim = 3
        self.dn1 = nn.Linear(dec_size+hidden_dim, 100)
        self.dn2 = nn.Linear(100, 50)
        self.dn3 = nn.Linear(50, self.output_dim)

    def forward(self, x, properties):
        inp = x.clone()
        #prop = properties.clone()
        # Handle various batch sizes between regular state vs. transition history - not very elegant, but it works
        if len(inp.shape) == 1:
            inp.unsqueeze_(0).unsqueeze_(0)  # Cast to shape [1,1,input_dim] - needed for LSTM
        else:
            inp.unsqueeze_(0)
        self.out, self.hidden = self.lstm_layer(inp, self.hidden)
        #x.squeeze(0).squeeze(0)

        legal = F.relu(self.ln1(properties))
        legal = F.relu(self.ln2(legal))

        # Decision network
        if self.out.shape[1] == 1:
            ind = torch.cat((legal, self.out.squeeze(0).squeeze(0)),0).to(device)  # Squeeze out tensor to recast to [hidden_dim]
        else:
            ind = torch.cat((self.out, legal.unsqueeze(0)),2).squeeze(0).to(device)  # Cat batches
        dec = F.relu(self.dn1(ind))
        dec = F.relu(self.dn2(dec))

        return self.dn3(dec)

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
        #state = torch.cat((torch.tensor([asset_status]).type('torch.FloatTensor'), state), 0).to(device)
        #state = torch.cat((torch.tensor([bought]).type('torch.FloatTensor'), state), 0).to(device)
        #state = torch.cat((torch.tensor([hold_time]).type('torch.FloatTensor'), state), 0).to(device)
        return state

    def normalize(self):
        # Normalize train_env/ask to [-1,1]
        self.spread = max(self.train_ask) - min(self.train_ask)
        self.offset = min(self.train_ask)
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


class Agent(object):

    def __init__(self, policy, state_size):

        # Model parameters
        self.input_dimension = state_size
        self.n_actions = 3  # Buy, hold, sell

        # State parameters
        self.state = None
        self.asset_status = 0
        self.bought = 0
        self.hold_time = 0
        self.gain_track = 1

        # Memory
        self.mem_capacity = 10000 
        self.memory = ReplayMemory(self.mem_capacity)

        # Financial parameters
        self.fee = 0.00075  # 0.075% for Binance
        self.investment = 1#000
        self.investment_scale = 1000  # Some numerical issues if actual investment is this high, so just scale what we report
        self.losses = []
        self.gains = []
        self.episode_profit = 0
        self.profit_track = []
        self.initial_market_value = 0
        self.session_begin_value = 0
        self.gain_buy_sell = []
        self.loss_buy_sell = []
        self.gain_hold = []
        self.loss_hold = []

        if policy == 'mlp':
            self.policy_net = DQN(self.input_dimension, self.n_actions).to(device)
            self.target_net = DQN(self.input_dimension, self.n_actions).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
        if policy == 'lstm':
            self.n_layers = 1
            self.policy_net = LSTM(self.input_dimension, self.n_actions, self.n_layers, self.gen_properties()).to(device)
            self.target_net = LSTM(self.input_dimension, self.n_actions, self.n_layers, self.gen_properties()).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

        # Learning parameters
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.001
        self.EPS_DECAY = 5000
        self.TARGET_UPDATE = 80
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.total_steps = 0
        self.BATCH_SIZE = 128
        self.hold_penalty = 10 #np.inf  # Encourage the model to trade more quickly (hold_time/hold_penalty)


    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def report(self, ask, spread, offset):
        if len(self.gains) >= 20:

            print("\nGlobal start: ${}, current: :${}".format(round(self.initial_market_value*spread + offset, 2), ask*spread + offset))
            print("Market moved ${} over the session".format(round((ask*spread + offset) - self.session_begin_value,2)))
            print("Start: ${}, current: ${}".format(round(self.initial_market_value*spread + offset,3), round(ask*spread + offset,3)))
            print("     Session wins: {} @ $ {}, avg hold: {} steps".format(len(self.gains), self.investment_scale*round(sum(self.gains)/len(self.gains),3), round(sum(self.gain_hold)/len(self.gain_hold),0)))
            print("     Session loss: {} @ ${}, avg hold: {} steps".format(len(self.losses), self.investment_scale*round(sum(self.losses)/len(self.losses),3), round(sum(self.loss_hold)/len(self.loss_hold),0)))
            print("     Session Net:  ${}".format(round(self.investment_scale*(sum(self.gains) + sum(self.losses)),2)))  # Losses are already negative
            print("     Episode total: ${}\n\n\n".format(round(self.investment_scale*self.episode_profit,2)))

            self.session_begin_value = ask*spread + offset
            self.gains = []
            self.losses = []
            self.gain_hold = []
            self.loss_hold = []

    def gen_properties(self):
        return torch.tensor([self.asset_status, self.bought, self.hold_time]).type('torch.FloatTensor')


    # Return asset status, bought value, 
    def take_action(self, state, properties, ask, next_env, next_ask):
        
        rand = random.random()
        epsilon_threshold = self.EPS_END + (self.EPS_START - self.EPS_END)*math.exp(-1. * self.total_steps/self.EPS_DECAY)
        self.total_steps += 1
        if rand > epsilon_threshold:
            with torch.no_grad():
                action = self.policy_net(state, properties).max(0)[1].view(1,1)  # Returns the index of the maximum output in a 1x1 tensor
        else:
            #print("Chose randomly ({})...".format(epsilon_threshold))
            action = torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

        
        reward = self.reward_calc(action, self.asset_status, self.investment, self.bought, self.hold_time, self.fee, ask)

        # Calculate target values
        target = next_env
        # Add the current asset status and bought value to the next
        #target = torch.cat((torch.tensor([self.asset_status]).type('torch.FloatTensor'), target), 0).to(device)
        #target = torch.cat((torch.tensor([self.bought]).type('torch.FloatTensor'), target), 0).to(device)
        #target = torch.cat((torch.tensor([self.hold_time]).type('torch.FloatTensor'), target), 0).to(device)

        # Push memory into buffer
        next_properties = self.gen_properties()
        next_action = 0
        next_reward = 0
        with torch.no_grad():
            sold_target = next_ask
            next_action = self.target_net(target, next_properties).max(0)[1].view(1,1)

            # All local values were updated with the last reward_calc
            next_reward = self.target_reward_calc(next_action, self.asset_status, self.investment, self.bought, self.hold_time, self.fee, next_ask)

        
        self.memory.push(state, properties, action, reward, target, next_reward)

    # This separate calculator exists so we don't update the agen't state with the target reward calculation. Not ideal...
    def target_reward_calc(self, action, asset_status, investment, bought, hold_time, fee, ask):

        reward = 0
        # Money is in our wallet
        if self.asset_status == 0:

            # Buying is legal
            if action == 0:
                reward = 0

            # Holding is legal, but don't hold forever
            if action == 1:
                reward = -self.hold_time/self.hold_penalty

            # Selling is illegal
            if action == 2:
                reward = -100

        # Money is in an asset
        if self.asset_status == 1:

            # Buying is illegal
            if action == 0:
                reward = -100

            # Holding is legal, but don't hold forever
            if action == 1:
                reward = -self.hold_time/self.hold_penalty

            # Selling is legal
            if action == 2:
                buy_cost = self.investment*self.fee
                bought_shares = (self.investment - buy_cost)/self.bought
                sold_revenue = bought_shares*ask
                sell_cost = sold_revenue*self.fee
                reward = sold_revenue - sell_cost - buy_cost - self.investment
        
        return torch.tensor([reward]).type('torch.FloatTensor')

    ### Try incentivising it to sell sooner. It's starting to hold for a long time
    ### Try making the reward the total value, not just the immediate value
    def reward_calc(self, action, asset_status, investment, bought, hold_time, fee, ask):

        # Money is in our wallet
        if self.asset_status == 0:

            # Buying is legal
            if action == 0:
                self.bought = ask
                self.asset_status = 1  # Money is now in the asset
                reward = 0

            # Holding is legal, but don't hold forever
            if action == 1:
                self.hold_time += 1
                reward = -self.hold_time/self.hold_penalty

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

            # Holding is legal, but don't hold forever
            if action == 1:
                self.hold_time += 1
                reward = -self.hold_time/self.hold_penalty

            # Selling is legal
            if action == 2:
                self.asset_status = 0  # Money is back in our wallet
                

                buy_cost = self.investment*self.fee
                bought_shares = (self.investment - buy_cost)/self.bought
                sold_revenue = bought_shares*ask
                sell_cost = sold_revenue*self.fee
                reward = sold_revenue - sell_cost - buy_cost - self.investment
                self.episode_profit += reward

                if reward > 0:
                    print("Gain: $ {}, bought at {}, sold at {} after {} steps".format(round(self.investment_scale*reward,2), round(self.bought,4), round(ask,4), self.hold_time))
                    self.gain_track += 1
                    self.gains.append(reward)
                    self.gain_buy_sell.append((round(self.bought,4), round(ask,4)))
                    self.gain_hold.append(self.hold_time)
                    self.profit_track.append(reward)

                if reward <= 0:
                    print("Loss: ${}, bought at {}, sold at {} after {} steps".format(round(self.investment_scale*reward,2), round(self.bought,4), round(ask,4), self.hold_time))
                    self.losses.append(reward)
                    self.loss_buy_sell.append((round(self.bought,4), round(ask,4)))
                    self.loss_hold.append(self.hold_time)
                    self.profit_track.append(reward)

                #if self.episode_profit > 0:
                #    reward = self.episode_profit

                self.hold_time = 0

        return torch.tensor([reward]).type('torch.FloatTensor')

    def optimize_model(self, BATCH_SIZE):

        if self.memory._len() < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)

        batch_state = []
        batch_prop = []
        batch_action = []
        batch_reward = []
        batch_target = []
        batch_next_action = []
        for i, trans in enumerate(transitions):
            batch_state.append(trans.state)
            batch_prop.append(trans.properties)
            batch_action.append(trans.action[0])
            batch_reward.append(trans.reward[0])
            batch_target.append(trans.target[0])
            batch_next_action.append(trans.next_action[0])

        batch_state = torch.stack(batch_state).to(device)
        batch_prop = torch.stack(batch_prop).to(device)
        batch_action = torch.stack(batch_action).to(device)
        batch_reward = torch.stack(batch_reward).to(device)
        batch_target = torch.stack(batch_target).to(device)
        batch_next_action = torch.stack(batch_next_action).to(device)

        # State-action values
        state_action_values = self.policy_net(batch_state, batch_prop).gather(1, batch_action)

        # Target values
        expected_state_action_values = (batch_next_action[0]*self.GAMMA) + batch_reward

        #loss = F.smooth_l1_loss(state_action_values, batch_reward.unsqueeze(1))
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
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

        # Initialize the agent
        self.agent = Agent('lstm', self.env.train_env[0].size()[0])
        self.agent.initial_market_value = self.env.train_ask[0]

    def trade(self):
        # Iterate over states

        for episode in range(self.env.num_episodes):
            for i, state in enumerate(self.env.train_env):

                # Add env parameters to the state
                state = self.env.update(state, self.agent.asset_status, self.agent.bought, self.agent.hold_time)
                
                # Take an action
                self.agent.take_action(state, self.agent.gen_properties(), self.env.train_ask[i], self.env.train_env[i+1], self.env.train_ask[i+1])

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


# Add some parameters here so it's easier to prototype
investment = 1
model = 'lstm'

beast = execute()
beast.trade()

