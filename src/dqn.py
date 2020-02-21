
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
        self.extra_values = 3  # Asset status, bought value, hold_time
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
        state = torch.cat((state, torch.tensor([asset_status]).type('torch.FloatTensor')), 0)
        state = torch.cat((state, torch.tensor([bought]).type('torch.FloatTensor')), 0)
        state = torch.cat((state, torch.tensor([hold_time]).type('torch.FloatTensor')), 0).to(device)
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

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def buy(self):
        raise NotImplementedError

    def hold(self):
        raise NotImplementedError

    def sell(self):
        raise NotImplementedError

    def report(self, ask, spread, offset):
        if len(self.gains) >= 20:
            print(self.gains)
            print(" ")
            print(self.losses)
            print(" ")
            print("Start: ${}, current: ${}".format(round(self.initial_market_value*spread + offset,3), round(ask*spread + offset,3)))
            print("     Session wins: {} @ ${}".format(len(self.gains), len(self.gains)/sum(self.gains)))
            print("     Session loss: {} @ ${}".format(len(self.losses), len(self.losses)/sum(self.losses)))
            print("     Session Net:  ${}".format(sum(self.gains) - sum(self.losses)))
            print("     Episode total: ${}\n\n\n".format(self.episode_profit))



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
        target = torch.cat((target, torch.tensor([self.asset_status]).type('torch.FloatTensor')), 0)
        target = torch.cat((target, torch.tensor([self.bought]).type('torch.FloatTensor')), 0)
        target = torch.cat((target, torch.tensor([self.hold_time]).type('torch.FloatTensor')), 0).to(device)

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
                self.sold = ask
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
        
                # Track total profit for legal trades
                self.episode_profit += reward

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
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()


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
                if self.agent.gain_track % self.agent.TARGET_UPDATE == 0:
                    self.agent.gain_track += 1
                    print("Updating target net...")
                    self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
        
        # Save policy every episode
        torch.save(self.agent.policy_net.state_dict(),"saved_policy_{}_{}.pt".format('ETH',datetime.datetime.now()))



        # update target network

        # checkpoint model

        # 

        print("here")


beast = execute()
beast.trade()



































# Replay memory class


#################################################################
### Q-Networks | may put this in another file
#################################################################
'''
class DQN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_dim, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
'''

# Other possible networks


class CNN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                                 dilation=1, return_indices=False,
                                 ceil_mode=False)

        self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
        #self.conv_attention = nn.Conv1d(n_hid, 1, kernel_size=1)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        # Input shape: [num_sims * num_edges, num_dims, num_timesteps]

        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        pred = self.conv_predict(x)
        #attention = my_softmax(self.conv_attention(x), axis=2)

        #edge_prob = (pred * attention).mean(dim=2)
        return pred

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=3,
                    num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)


#################################################################
### Data Pre-Processing | may put this in another file
#################################################################

# Returns two values
# 1) train_environment: a list train_length*#params long corresponding to a history of 
# train_length at each period interval
# 2) env_value: a list of floats corresponding to the ask value at the current timestep
# TODO: only list current timestep, not all
def parse_data(path, train_length, params, period):
    print("Processing data...")
    with open(path, 'r') as f:
        data = csv.reader(f)
        data = list(data)
        labels = data.pop(0)
        indices = [labels.index(i) for i in params]
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
    #train_environment = train_environment.double()
    print(train_environment)

    env_value = np.array(env_value)
    env_value = env_value.astype(float)

    return train_environment, env_value

#################################################################
### Utilities | may put this in another file
#################################################################

def select_action(state):
    global total_steps
    rand = random.random()
    epsilon_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1. * total_steps/EPS_DECAY)
    total_steps += 1
    #print("eps thresh: ", epsilon_threshold)
    if rand > epsilon_threshold:
        with torch.no_grad():
            return policy_net(state).max(0)[1].view(1,1)  # Returns the index of the maximum output in a 1x1 tensor
    else:
        #print("Chose randomly ({})...".format(epsilon_threshold))
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


# Incentivize legal/illegal moves appropriately
# Actions: {0:buy, 1:hold, 2:sell}
def reward_calc(action, asset_status, investment, bought, sold, hold_time, fee, scale):

    action = action.item()
    #print("Asset status: ", asset_status, ", action: ", action)

    # Our money is still in our wallet
    if asset_status == 0:
        if action == 0:
            #print("\n\nBought appropriately at ${}".format(bought))
            reward = 0
        if action == 1:
            #print("Hold")
            reward = 0
        # Can't sell assets we don't have
        if action == 2:
            #print("Whoops, can't sell assets we don't have")
            reward = -100

    # Our money is out in the asset
    if asset_status == 1:
        # Can't buy assets without any cash
        if action == 0:
            #print("Whoops, can't buy assets without cash")
            reward = -100

        if action == 1:
            #print("Hold")
            reward = 0

        if action == 2:
            #print("Sold appropriately at ${} after {} steps".format(sold, hold_time))
            #print("Invested $", investment, " at $", bought, ", sold at $", sold, " with fee = ", fee)
            #print("*"*30)

            buy_cost = investment*fee
            bought_shares = investment/bought
            sold_revenue = bought_shares*sold
            sold_cost = sold_revenue*fee

            reward = investment*sold/bought - investment - investment*fee - investment*(sold/bought)*fee
            
            # Increase reward for gains, keep the same for losses
            #if reward > 0:
            reward *= scale
            #print("$", reward/scale, " bought: ", bought, ", sold: ", sold, ", after: ", hold_time)
            #print("         Gross {} with {} total fees | {}/{}".format(sold_revenue - investment, buy_cost+sold_cost, bought, sold))

    return torch.tensor([reward]).type('torch.FloatTensor')

#################################################################
### Model optimization | may put this in another file
#################################################################

def optimize_model():
    if memory._len() < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    batch_state = []
    batch_action = []
    batch_reward = []
    batch_target = []
    batch_next_action = []
    for i, trans in enumerate(transitions):
        batch_state.append(trans.state)
        batch_action.append(trans.action[0])
        print("Reg trans reward")
        print(trans.reward)
        batch_reward.append(trans.reward[0])
        batch_target.append(trans.target[0])
        batch_next_action.append(trans.next_action[0])

    batch_state = torch.stack(batch_state).to(device)
    batch_action = torch.stack(batch_action).to(device)
    batch_reward = torch.stack(batch_reward).to(device)
    batch_target = torch.stack(batch_target).to(device)
    batch_next_action = torch.stack(batch_next_action).to(device)

    # Add target here

    

    state_action_values = policy_net(batch_state).gather(1, batch_action)

    expected_state_action_values = batch_next_action
    expected_state_action_values = (batch_next_action[0]*GAMMA) + batch_reward

    #print("State action values: \n")
    #print(state_action_values)

    #print("\n Batch_reward: \n")
    #print(expected_state_action_values.unsqueeze(1))

    #loss = F.smooth_l1_loss(state_action_values, batch_reward.unsqueeze(1))
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1,1)
    optimizer.step()


#################################################################
### Model training | may put this in another file
#################################################################

# Update batch size
# Normalize inputs
# Print wins, see if that is happening here



BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.001
EPS_DECAY = 5000
TARGET_UPDATE = 80

data_path = '/Users/mattsonthieme/Documents/Academic.nosync/Projects/RLTrading/data/crypto/ETH/ETH_1s_4.csv'

minutes_back = 60 

train_period = 15  # Seconds between market checks
train_length = 240 #minutes_back*60/train_period
train_params = ['ask']
n_actions = 3  # Buy, hold, sell
mem_capacity = 10000

# Initial investment, will be adjusted as we proceed
investment = 1000

extra_values = 3  # Asset status, bought value, hold_time

scale = 1  # Scale reward values

input_dimension = train_length*len(train_params) + extra_values
output_dimension = n_actions

policy_net = DQN(input_dimension, output_dimension).to(device)
target_net = DQN(input_dimension, output_dimension).to(device)
#policy_net = CNN(1, 128, output_dimension, do_prob=0.).to(device)
#target_net = CNN(1, 128, output_dimension, do_prob=0.).to(device)
#policy_net = LSTM(1, 500, batch_size=1, output_dim=output_dimension, num_layers=2).to(device)
#target_net = LSTM(1, 500, batch_size=1, output_dim=output_dimension, num_layers=2).to(device)


target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
#optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

memory = ReplayMemory(mem_capacity)

total_steps = 0

num_episodes = 1



#train_env, train_ask = parse_data(data_path, train_length, train_params, train_period)
#train_env = np.load('highres13hrs_env.npy')
#train_ask = np.load('highres13hrs_ask.npy')

#torch.save(train_env, 'ETH_ask_p15s_l10m_env.pt')
#torch.save(train_ask, 'ETH_ask_p15s_l10m_ask.pt')


train_env = torch.load('highres3days_p15s_l60m_env.pt')
train_ask = torch.load('highres3days_p15s_l60m_ask.pt')

#train_env = torch.load('ETH_ask_p15s_l60m_env.pt')
#train_ask = torch.load('ETH_ask_p15s_l60m_ask.pt')


# Normalize train_env [-1,1]
train_env = (train_env - train_ask.mean().item())/(max(train_ask) - min(train_ask))


print(train_env)
#print(max(train_ask))
#print(train_env/max(train_ask))

for i_episode in range(num_episodes):
    # Initialize environment
    
    train_size = len(train_env)


    ### Try other periods
    ### Try an LSTM 
    ### Try having the LSTM predict whether the next step will be up or down, need a new dataset
    ### What do most successful trades look like? How long do we wait? 
    ### Consider adding a requirement that the trade makes money
    ### Add learning rate to optim
    ### Figure out what param.grad.data.clamp_(-1,1) does



    # Track bought and sold prices
    bought = 0
    sold = 0
    asset_status = 0
    fee = 0.0 #0.00075
    hold_time = 0
    reward_track = 0
    episode_profit = 0
    target_update_counter = 0

    market_start = train_ask[0]
    session_start = 0
    start_ask = train_ask[0]

    rolling_track = 20

    profits = []
    losses = []

    for i, state in enumerate(train_env[:train_size-1]):
        print("start: ", bought, ", asset status: ", asset_status)

        if (len(profits)%rolling_track) == 0 and len(profits) > 0:
            print("\n\n")
            print("(Global start: ${}, current ${})".format(start_ask, train_ask[i]))
            print("Last {} trades in {} minutes:".format(len(profits) + len(losses), train_period*(i - session_start)/60))
            session_start = i
            print("   Market moved:    ${}".format(round(train_ask[i] - market_start,2)))
            market_start = train_ask[i]
            print("   {} gains, avg:   ${}".format(len(profits), round(sum(profits)/len(profits),2)))
            print("   {} losses, avg:  ${}".format(len(losses), round(sum(losses)/len(losses),2)))
            print("   Session Net:     ${}".format(round(sum(profits) + sum(losses),2)))  # Losses already negative
            episode_profit += sum(profits) + sum(losses)
            print("   Episode Net:     ${}".format(round(episode_profit,2)))
            print("\n\n")
            profits = []
            losses = []
        
        if i%(int(train_size/10)) == 0:
            print("\n\n\n\n\n\n")
            print("#"*60)
            print("{}% complete with episode {}/{}".format(int(100*float(i)/train_size), i_episode, num_episodes))
            print("#"*60)
            print("\n\n\n\n\n\n")

        # Add the current asset status and bought value to the state to the state
        state = torch.cat((state, torch.tensor([asset_status]).type('torch.FloatTensor')), 0)
        state = torch.cat((state, torch.tensor([bought]).type('torch.FloatTensor')), 0)
        state = torch.cat((state, torch.tensor([hold_time]).type('torch.FloatTensor')), 0).to(device)

        # View current state, select action
        action = select_action(state)

        #print("\n\nAsset status: {}, bought: {}, ask:{}, hold_time: {}, action: {}".format(asset_status, bought, train_ask[i], hold_time, action))

        ## Buy
        if action == 0:
            
            # Only allow us to buy when we're holding cash
            if asset_status == 0:
                bought = train_ask[i]
                reward = reward_calc(action, asset_status, investment, bought, sold, hold_time, fee, scale)

                asset_status = 1
                hold_time = 0
            else:
                asset_status = 1
                hold_time += 1
                reward = reward_calc(action, asset_status, investment, bought, sold, hold_time, fee, scale)

        ## Hold
        if action == 1:
            hold_time += 1
            reward = reward_calc(action, asset_status, investment, bought, sold, hold_time, fee, scale)

        ## Sell
        if action == 2:

            sold = train_ask[i]
            # Only allow us to sell when we're holding assets
            if asset_status == 1:
                reward = reward_calc(action, asset_status, investment, bought, sold, hold_time, fee, scale)
                if reward > 0:
                    #print("i = ",i, " held for ", hold_time)
                    reward_track += reward.item()/scale
                    profits.append(reward.item()/scale)
                    target_update_counter += 1
                else:
                    #print("i = ",i, " held for ", hold_time)
                    reward_track += reward.item()/scale
                    losses.append(reward.item()/scale)

                hold_time = 0
                asset_status = 0
            else:
                asset_status = 0
                hold_time += 1
                reward = reward_calc(action, asset_status, investment, bought, sold, hold_time, fee, scale)

            bought = 0
        #print("finish: ", bought, ", asset status: ", asset_status)

        target = train_env[i+1]
        # Add the current asset status and bought value to the state to the state
        target = torch.cat((target, torch.tensor([asset_status]).type('torch.FloatTensor')), 0)
        target = torch.cat((target, torch.tensor([bought]).type('torch.FloatTensor')), 0)
        target = torch.cat((target, torch.tensor([hold_time]).type('torch.FloatTensor')), 0).to(device)

        next_action = 0
        next_reward = 0
        with torch.no_grad():
            sold_target = train_ask[i+1]
            next_action = target_net(target).max(0)[1].view(1,1)
            next_reward = reward_calc(next_action, asset_status, investment, bought, sold_target, hold_time, fee, scale)

        memory.push(state, action, reward, target, next_reward)


        optimize_model()


        if target_update_counter % TARGET_UPDATE == 0:
            target_update_counter += 1
            print("Updating target net...")
            target_net.load_state_dict(policy_net.state_dict())

    print("Net: $", reward_track)

    torch.save(policy_net.state_dict(),"saved_policy_{}_{}.pt".format('ETH',datetime.datetime.now()))

























#parse_data(data_path, train_length, train_params, train_period)


#################################################################
### Plotting | may put this in another file
#################################################################

'''

with open(data_path, 'r') as f:
    data = csv.reader(f)
    data = list(data)
    labels = data.pop(0)
    #indices = [labels.index(i) for i in params]
    data = np.array(data)
    bidVolume = []
    ask = []
    askVolume = []

    time = []
    to_plot = []
    bv = []
    av = []
    vwap = []

    for row in data:
        time.append(float(row[0]))
        to_plot.append(float(row[5]))
        av.append(float(row[7]))


    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Ask', color=color)
    ax1.plot(time, to_plot, color=color)
    #ax1.plot(time, vwap, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor=color)


    #ax2 = ax1.twinx()
    #color = 'tab:blue'
    #ax2.set_ylabel('askVolume', color=color)  # we already handled the x-label with ax1
    #ax2.plot(time, av, color=color)
    #ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
     
''' 















