
## TODO ##
# Point to appropriate data folder
# Store 

import math
import random
import time
import numpy as np
from collections import namedtuple
import csv
import configuration

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transition states for updating the model
Transition = namedtuple('Transition',
                        ('state', 'properties', 'action', 'reward', 'target', 'next_action'))

# This model has three parts
# 1) An LSTM network processing only price history data
# 2) A 'legality network' looking at asset_status, bought price, hold_time to determine if certain actions are legal
# 3) A 'decision network' looking at the output of 1) and 2) to make the final decision
class MultiPhase(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, properties):
        super(MultiPhase, self).__init__()
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
        # Haven't included dropout here because properties are deterministic and not noisy
        prop_length = properties.size()[0]
        dec_size = 8  # Size of the final embedding of the legality network, can try other values
        self.ln1 = nn.Linear(prop_length, 128)
        self.LLn1 = nn.LayerNorm(128)
        self.ln2 = nn.Linear(128, 64)
        self.LLn2 = nn.LayerNorm(64)
        self.ln3 = nn.Linear(64, dec_size)

        # Decision network
        self.output_dim = output_dim
        self.dn1 = nn.Linear(dec_size+hidden_dim, 500)
        self.DLn1 = nn.LayerNorm(500)
        self.dn2 = nn.Linear(500, 200)
        self.DLn2 = nn.LayerNorm(200)
        self.dn3 = nn.Linear(200, 50)
        self.DLn3 = nn.LayerNorm(50)
        self.dn4 = nn.Linear(50, self.output_dim)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, properties):
        inp = Variable(x)
        properties = Variable(properties)

        # Sequence network
        self.out, self.hidden = self.lstm_layer(inp, self.hidden)

        # Legality network
        legal = F.relu(self.LLn1(self.ln1(properties)))
        legal = F.relu(self.LLn2(self.ln2(legal)))
        legal = self.ln3(legal)

        # Decision network - handle batches
        if self.out.shape[1] == 1:
            ind = torch.cat((legal, self.out.squeeze(0).squeeze(0)),0).to(device)  # Squeeze out tensor to recast to [hidden_dim]
        else:
            ind = torch.cat((self.out, legal.unsqueeze(0)),2).squeeze(0).to(device)  # Cat batches
        
        dec = F.relu(self.DLn1(self.dn1(ind)))
        dec = F.relu(self.DLn2(self.dn2(dec)))
        dec = F.relu(self.DLn3(self.dn3(dec)))
        dec = self.dn4(dec)
        dec = F.softmax(dec, dim=0)

        return dec

# Simple MLP model to verify functionality
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

# Environment object which creates/stores datasets
class Env(object):

    def __init__(self, assets, minutes_back, period, params):
        self.data_path = None
        self.assets = assets
        self.minutes_back = configuration.minutes_back
        self.train_period = configuration.period
        self.train_length = self.minutes_back*60/self.train_period
        self.params = configuration.params

        # Initial environment parameters
        self.asset_status = 0
        self.bought = 0
        self.hold_time = 0

        self.num_episodes = 10  # Somewhat defunct with new data-loader, but will keep for now
        self.train_params = params  # ['ask']
        self.train_env = None
        self.train_ask = None

        # Normalization/conversion
        self.scale = 0
    
    def reset(self):
        # Initial environment parameters
        self.asset_status = 0
        self.bought = 0
        self.hold_time = 0

    def load_data(self):
        try:
            self.train_env = torch.load('../data/crypto/{}_ask_p{}s_l{}m_env.pt'.format(self.assets, self.train_period, self.minutes_back))
            self.train_ask = torch.load('../data/crypto/{}_ask_p{}s_l{}m_ask.pt'.format(self.assets, self.train_period, self.minutes_back))
            print("Loading {}_ask_p{}s_l{}m_env.pt".format(self.assets, self.train_period, self.minutes_back))
        except:
            print("Creating {}_ask_p{}s_l{}m_env.pt".format(self.assets, self.train_period, self.minutes_back))
            self.data_path = configuration.data_path
            self.train_env, self.train_ask = self.parse_data(self.data_path, self.train_length, self.train_params, self.train_period)
            torch.save(self.train_env, '../data/crypto/{}_ask_p{}s_l{}m_env.pt'.format(self.assets, self.train_period, self.minutes_back))
            torch.save(self.train_ask, '../data/crypto/{}_ask_p{}s_l{}m_ask.pt'.format(self.assets, self.train_period, self.minutes_back))

    def normalize(self):
        # Normalize train_env/ask to [0,1]
        max_val = 0
        for i in self.train_ask:
            max_val = max(max(i), max_val)

        print("Normalizing to {}".format(max_val))
        self.scale = max_val
        self.train_env = self.train_env/max_val
        self.train_ask = self.train_ask/max_val

    # Returns two values
    # 1) train_environment: a list train_length*#params long corresponding to a history of train_length at each period interval
    # 2) env_value: a list of floats corresponding to the ask value at the current timestep
    def parse_data(self, path, train_length, params, period):
        print("Processing data...")
        with open(path, 'r') as f:
            data = csv.reader(f)
            data = list(data)
            labels = data.pop(0)
            indices = [labels.index(i) for i in params]
            max_index = len(data)-len(data)%period  # Don't overshoot
            data = np.array(data[:max_index])

            train_environment = []
            env_value = []


            # Multiplex periods of > 1 second
            for i in range(period):

                train_episode = []
                val_episode = []

                # Filter data s.t. only rows corresponding to that period remain
                period_indices = [j for j in range(i, len(data), period)]
                tempdata = data[period_indices]

                # Swap col/rows for easier access 
                tempdata = tempdata.transpose()

                # Slice indices to determine individual environment states
                begin = 0
                end = int(train_length)
                
                while end < len(tempdata[0]):
                    train_elem = []
                    for j in indices:
                        train_elem.extend(tempdata[j][begin:end])
                    
                    train_episode.append(train_elem)
                    val_episode.append(tempdata[labels.index('ask')][end-1])  # Only looking at ask price for now

                    begin += 1
                    end += 1
                print("{}% complete...".format(int(100*float(i)/period)))

                train_environment.append(train_episode)
                env_value.append(val_episode)

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
        self.hidden_dim = 32
        self.n_actions = 3  # Buy, hold, sell

        # State parameters
        self.state = None
        self.asset_status = 0
        self.bought = 0
        self.hold_time = 0

        # Memory
        self.mem_capacity = configuration.mem_capacity
        self.memory = ReplayMemory(self.mem_capacity)

        # Financial parameters
        self.fee = configuration.fee  # 0.075% for Binance - 0% because Robinhood is on the scene!
        self.investment = 1
        self.investment_scale = configuration.investment_scale  # Some numerical issues if actual investment is this high, so just scale what we report
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
        self.last_ask = 0
        self.trade_cycle = []

        # Simple network for testing functionality
        if policy == 'mlp':
            self.policy_net = DQN(self.input_dimension, self.n_actions).to(device)
            self.target_net = DQN(self.input_dimension, self.n_actions).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
        
        # Actual network for trading
        if policy == 'multiphase':
            self.n_layers = 1
            self.policy_net = MultiPhase(self.input_dimension, self.hidden_dim, self.n_actions, self.n_layers, self.gen_properties()).to(device)
            self.target_net = MultiPhase(self.input_dimension, self.hidden_dim, self.n_actions, self.n_layers, self.gen_properties()).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

        # Learning parameters
        self.GAMMA = configuration.GAMMA
        self.EPS_START = configuration.EPS_START
        self.EPS_END = configuration.EPS_END
        self.EPS_DECAY = configuration.EPS_DECAY  # Increasing in the hopes that it will help the model learn more about short term opportunities - takes ~80k steps to get to its lower bound
        self.TARGET_UPDATE = configuration.TARGET_UPDATE
        self.POLICY_UPDATE = configuration.POLICY_UPDATE  # Will update this actively in report (for now)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        #self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  # Another optimizer - not sure which is optimal yet
        self.total_steps = configuration.total_steps
        self.BATCH_SIZE = configuration.BATCH_SIZE

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # Report wins/losses every 20 wins
    def report(self, ask, scale, step, total, last):
        if (len(self.gains) >= 20) or (last):

            print("\nGlobal start: ${}, current: :${}  -- ({}/{})".format(round(self.initial_market_value*scale, 2), ask*scale, step, total))
            print("Market moved ${} over the session".format(round((ask*scale) - self.session_begin_value,2)))
            print("Start: ${}, current: ${}".format(round(self.initial_market_value*scale,3), round(ask*scale,3)))
            print("     Session wins: {} @ $ {}, avg hold: {} steps".format(len(self.gains), self.investment_scale*round(sum(self.gains)/(len(self.gains)+1),5), round(sum(self.gain_hold)/(len(self.gain_hold)+1),0)))
            print("     Session loss: {} @ ${}, avg hold: {} steps".format(len(self.losses), self.investment_scale*round(sum(self.losses)/(len(self.losses)+1),5), round(sum(self.loss_hold)/(len(self.loss_hold)+1),0)))
            print("     Session Net:  ${}".format(round(self.investment_scale*(sum(self.gains) + sum(self.losses)),2)))  # Losses are already negative
            print("     Episode total: ${}\n\n\n".format(round(self.investment_scale*self.episode_profit,2)))

            self.session_begin_value = ask*scale
            self.gains = []
            self.losses = []
            self.gain_hold = []
            self.loss_hold = []

    # Generate new properties after each decision
    def gen_properties(self):
        return torch.tensor([self.asset_status, self.bought, 2.0/(self.hold_time+1)-1]).type('torch.FloatTensor')  # 2/self.hold_time because I think the enormous numbers are throwing off the L-network. Scales to [1, -1]

    # Reset environment each episode
    def reset(self):
        # State parameters
        self.asset_status = 0
        self.bought = 0
        self.hold_time = 0

        # Financial parameters
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

        self.testA = None
        self.testB = None

    # Return asset status, bought value, 
    def take_action(self, state, properties, last_ask, ask, next_env, next_ask):
        rand = random.random()
        epsilon_threshold = self.EPS_END + (self.EPS_START - self.EPS_END)*math.exp(-1. * self.total_steps/self.EPS_DECAY)

        self.total_steps += 1

        if self.total_steps % self.POLICY_UPDATE == 0:
            print("Epsilon threshold: ", epsilon_threshold)

        if rand > epsilon_threshold:
            with torch.no_grad():
                action = self.policy_net(state, properties).max(0)[1].view(1,1)  # Returns the index of the maximum output in a 1x1 tensor

        else:
            action = torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

        reward = self.reward_calc(action, self.asset_status, self.investment, self.bought, self.hold_time, self.fee, ask, target_calc=False)

        # Calculate target values
        target = next_env

        # Push memory into buffer
        next_properties = self.gen_properties()
        next_action = 0
        next_reward = 0
        with torch.no_grad():
            sold_target = next_ask
            next_action = self.target_net(target, next_properties).max(0)[1].view(1,1)

            # All local values were updated with the last reward_calc
            next_reward = self.reward_calc(next_action, self.asset_status, self.investment, self.bought, self.hold_time, self.fee, next_ask, target_calc=True)

        # Append trade to 
        self.trade_cycle.append([state, properties, action, reward, target, next_reward])

        ##### Assign all past actions to the value of this sale #####

        if action == 2:
            self.assign_rewards(reward)

    # Assign rewards to every decision leading up to this purchase
    def assign_rewards(self, reward):
        temp_reward = 0
        already_bought = False
        for decision in self.trade_cycle:
            state_ = decision[0]
            properties = decision[1]
            action = decision[2]

            if action == 0 and decision[3] != -1:
                already_bought = True

            if decision[3] == -1:
                temp_reward = decision[3]  # Punish illegal actions (keep existing reward)
            else:
                if reward > 0:
                    temp_reward = reward/(1 + self.hold_time/50)  # Current reward from this trade
                else:
                    temp_reward = reward

            target = decision[4]
            next_reward = decision[5]

            # Push new state/action/target into memory
            self.memory.push(state_, properties, action, temp_reward, target, next_reward)

        self.hold_time = 0
        self.trade_cycle = []

    def profit(self, ask):
        buy_cost = self.investment*self.fee
        bought_shares = (self.investment - buy_cost)/self.bought
        sold_revenue = bought_shares*ask
        sell_cost = sold_revenue*self.fee
        reward = sold_revenue - sell_cost - buy_cost - self.investment
        return reward

    def reward_calc(self, action, asset_status, investment, bought, hold_time, fee, ask, target_calc):

        # Money is in our wallet
        if self.asset_status == 0:

            # Buying is legal
            if action == 0:
                if not target_calc:
                    self.bought = ask
                    self.asset_status = 1  # Money is now in the asset
                reward = 0

            # Holding is legal, but don't hold forever
            if action == 1:
                if not target_calc:
                    self.hold_time += 1
                reward = 0

            # Selling is illegal
            if action == 2:
                if not target_calc:
                    self.hold_time += 1
                reward = -1

        # Money is in an asset
        if self.asset_status == 1:

            # Buying is illegal
            if action == 0:
                if not target_calc:
                    self.hold_time += 1
                reward = -1

            # Holding is legal, but don't hold forever
            if action == 1:
                if not target_calc:
                    self.hold_time += 1
                reward = 0

            # Selling is legal
            if action == 2:

                if not target_calc:
                    self.asset_status = 0  # Money is back in our wallet
                
                reward = self.profit(ask)

                if not target_calc:
                    self.episode_profit += reward

                # Update trackers
                if reward > 0 and not target_calc:
                    self.gains.append(reward)
                    self.gain_buy_sell.append((round(self.bought,4), round(ask,4)))
                    self.gain_hold.append(self.hold_time)
                    self.profit_track.append(reward)
                if reward <= 0 and not target_calc:
                    self.losses.append(reward)
                    self.loss_buy_sell.append((round(self.bought,4), round(ask,4)))
                    self.loss_hold.append(self.hold_time)
                    self.profit_track.append(reward)

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

        batch_state = torch.cat(batch_state).squeeze(1).unsqueeze_(0).to(device)
        batch_prop = torch.stack(batch_prop).to(device)
        batch_action = torch.stack(batch_action).to(device)
        batch_reward = torch.stack(batch_reward).to(device)
        batch_target = torch.cat(batch_target).squeeze(1).unsqueeze_(0).to(device)
        batch_next_action = torch.stack(batch_next_action).to(device)

        # State-action values
        state_action_values = self.policy_net(batch_state, batch_prop).gather(1, batch_action)

        # Target values
        expected_state_action_values = (batch_next_action[0]*self.GAMMA) + batch_reward

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)

        # Optional grad clamping - probably don't want to do this because some trades are better than others and we want the network updates to reflect that
        #for param in self.policy_net.parameters():  # restrict grad updates to (-1,1)
        #    param.grad.data.clamp_(-1,1)

        self.optimizer.step()

    # Implement to actually send orders with ccxt
    def buy(self):
        raise NotImplementedError

    def sell(self):
        raise NotImplementedError

# Executor class to initialize agent/env and begin trading/training
class execute(object):
    def __init__(self):

        # Params for initializing the environment
        self.asset = configuration.asset
        self.minutes_back = configuration.minutes_back
        self.period = configuration.period
        self.params = configuration.params

        # Initialize the environment
        self.env = Env(self.asset, self.minutes_back, self.period, self.params)
        self.env.load_data()
        self.env.normalize()

        # Initialize the agent
        state_size = self.env.train_env[0][0].size()[0]
        self.agent = Agent(configuration.model_type, state_size)

    def trade(self):

        for j in range(self.env.num_episodes):
            print("\n\n")
            print("#"*60)
            print("\n\nEpoch {}/{}\n\n".format(j, self.env.num_episodes))
            print("#"*60)
            print("\n\n")
            self.agent.total_steps = 0

            # Iterate over states
            for e, episode in enumerate(self.env.train_env):

                torch.save(self.agent.policy_net.state_dict(),"{}_policy_{}p_{}m.pt".format(self.asset, self.period, self.minutes_back))
                
                self.agent.last_ask = self.env.train_ask[e][0]

                # Reset environments with each episode
                self.agent.reset()
                self.env.reset()

                self.agent.initial_market_value = self.env.train_ask[0][0]
                
                # Iterate over all data
                for i, state in enumerate(episode[:episode.shape[0]-1]):

                    ask = self.env.train_ask[e][i]
                    next_ask = self.env.train_ask[e][i+1]
                    next_state = self.env.train_env[e][i+1]

                    if len(state.shape) == 1:
                        state = state.unsqueeze_(0).unsqueeze_(0)  # Cast to shape [1,1,input_dim] - needed for LSTM
                        next_state = next_state.unsqueeze_(0).unsqueeze_(0)
                    else:
                        state.unsqueeze_(0)
                        next_state.unsqueeze_(0)

                    # Take an action
                    self.agent.take_action(state, self.agent.gen_properties(), self.agent.last_ask, ask, next_state, next_ask)

                    # Optimize the agent according to that action
                    if i%self.agent.POLICY_UPDATE == 0:
                        print("Updating policy net...({}/{})".format(i, episode.shape[0]))
                        self.agent.optimize_model(self.agent.BATCH_SIZE)

                    # Output training info
                    self.agent.report(ask, self.env.scale, i, episode.shape[0], last=False)

                    # Update target network
                    if i%self.agent.TARGET_UPDATE == 0:

                        print("Updating target net...({}/{})".format(i, episode.shape[0]))
                        self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
                
                    self.agent.last_ask = ask

                # Output training info
                self.agent.report(self.env.train_ask[e][i], self.env.scale, i, episode.shape[0], last=True)  # Show the final result at the end of the episode
                print("#"*30)
                print("\nCompleted episode {} of {}\n".format(e, self.env.train_env.shape[0]))
                print("#"*30)

                # Save policy every episode
                torch.save(self.agent.policy_net.state_dict(),"{}_policy_{}p_{}m.pt".format(self.asset, self.period, self.minutes_back))


beast = execute()
beast.trade()
