
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
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x))

#################################################################
### Data Pre-Processing | may put this in another file
#################################################################

data_path = '/Users/mattsonthieme/Documents/Academic.nosync/Projects/RLTrading/data/highres0.csv'

def load_data(path, train_length, params, period):
    with open(path, 'r') as f:
        data = csv.reader(f)
        data = list(data)
        labels = data.pop(0)
        indices = [labels.index(i) for i in params]
        data = np.array(data)

        train_environment = []
        train_signal = []

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
                begin += 1
                end += 1

    print(len(train_environment[1]))
    print(train_environment[0])
    print(train_environment[1])
    print(train_environment[2])

    return 0

load_data(data_path, 10, ['bidVolume', 'ask', 'askVolume'], 10)










'''
    bidVolume = []
    ask = []
    askVolume = []

    time = []
    to_plot = []
    bv = []

    for row in data:
        time.append(float(row[0])/60)
        to_plot.append(float(row[6]))
        bv.append(float(row[5]))


    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Ask', color=color)
    ax1.plot(time, to_plot, color=color)
    ax1.tick_params(axis='y', labelcolor=color)


    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('bidVolume', color=color)  # we already handled the x-label with ax1
    ax2.plot(time, bv, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
        
'''
















