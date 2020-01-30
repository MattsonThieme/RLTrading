
'''
TODO

Parse data into ten datasets

Q Network

Simulation module
    Reward structure
    CCXT calls
        Test and real

Replay memory

Training loop

Visualize learning

'''

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Named tuple for storing transitions

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Replay memory class

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

    def _lenSelf(self):
        return len(self.memory)


#################################################################
### Q-Networks | may put this in another file
#################################################################

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

#################################################################
### Data Pre-Processing | may put this in another file
#################################################################

# Returns two values
# 1) train_environment: a list train_length*#params long corresponding to a history of 
# train_length at each period interval
# 2) env_value: a list of floats corresponding to the ask value at the current timestep
# TODO: only list current timestep, not all
def parse_data(path, train_length, params, period):
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

    train_environment = np.array(train_environment)
    train_environment = train_environment.astype(float)
    train_environment = torch.from_numpy(train_environment).type('torch.FloatTensor').to(device)
    #train_environment = train_environment.double()
    print(train_environment)

    env_value = np.array(env_value)
    env_value = env_value.astype(float)

    #print(len(train_environment[1]))
    #print(train_environment[0])
    #print(env_value[0])
    #print(train_environment[1])
    #print(train_environment[2])

    return train_environment, env_value

#################################################################
### Utilities | may put this in another file
#################################################################

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

data_path = '/Users/mattsonthieme/Documents/Academic.nosync/Projects/RLTrading/data/highres0.csv'
train_period = 15  # Seconds between market checks
train_length = 40
train_params = ['time', 'bidVolume', 'ask', 'askVolume']
n_actions = 3  # Buy, hold, sell
mem_capacity = 10000

input_dimension = train_length*len(train_params)
output_dimension = n_actions

policy_net = DQN(input_dimension, output_dimension).to(device)
target_net = DQN(input_dimension, output_dimension).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())

memory = ReplayMemory(mem_capacity)

total_steps = 0

def select_action(state):
    global total_steps
    rand = random.random()
    epsilon_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1. * total_steps/EPS_DECAY)
    total_steps += 1
    print("eps thresh: ", epsilon_threshold)
    if rand > epsilon_threshold:
        with torch.no_grad():
            return policy_net(state).max(0)[1].view(1,1)  # Returns the index of the maximum output in a 1x1 tensor
    else:
        print("Chose randomly...")
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


# Incentivize legal/illegal moves appropriately
# Actions: {0:buy, 1:hold, 2:sell}
def reward_calc(action, asset_status, investment, bought, sold, fee):

    scale = 10
    # Our money is still in our wallet
    if asset_status == 0:
        if action == 0:
            reward = 0
        if action == 1:
            reward = 0
        # Can't sell assets we don't have
        if action == 2:
            print("Whoops, can't sell assets we don't have")
            reward = -100

    # Our money is out in the asset
    if asset_status == 1:
        # Can't buy assets without any cash
        if action == 0:
            print("Whoops, can't buy assets without cash")
            reward = -100
        if action == 1:
            reward = 0
        if action == 2:
            reward = scale*((investment*sold - investment*sold*fee) - (investment*bought - investment*bought*fee))

    return reward

#################################################################
### Model optimization | may put this in another file
#################################################################

def optimize_model():
    if len(memory < BATCH_SIZE):
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition()


#################################################################
### Model training | may put this in another file
#################################################################


num_episodes = 1

for i_episode in range(num_episodes):
    # Initialize environment
    train_env, train_ask = parse_data(data_path, train_length, train_params, train_period)
    
    # Initial investment, will be adjusted as we proceed
    investment = 1000

    # Track bought and sold prices
    bought = 0
    sold = 0
    asset_status = 0
    fee = 0.00075

    for i, state in enumerate(train_env):
        print("i ", i)
        print(len(state))
        print(train_ask[i])

        action = select_action(state)
        print("action: ", action)
        if action == 0:
            bought = train_ask[i]
            asset_status = 1
            print("Bought at $", bought)
        if action == 2:
            sold = train_ask[i]
            investment = investment + reward_calc(action, asset_status, investment, bought, sold, fee)
            print("Sold at $", sold)
    


    print("New total = ", investment)

























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
        time.append(float(row[0])/60)
        to_plot.append(float(row[6]))
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

















