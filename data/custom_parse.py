import csv
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Returns two values
# 1) train_environment: a list train_length*#params long corresponding to a history of train_length at each period interval
# 2) env_value: a list of floats corresponding to the ask value at the current timestep
def parse_data(path, train_length, params, period, native_period):
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

        period = int(period/native_period)
        print("period ", period)

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
                val_episode.append(tempdata[labels.index(params[0])][end-1])

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

    train_environment, env_value = normalize(train_environment, env_value)

    return train_environment, env_value

def normalize(train_env, train_ask):
    # Normalize train_env/ask to [0,1]
    max_val = 0
    for i in train_ask:
        max_val = max(max(i), max_val)

    print("Normalizing to {}".format(max_val))
    scale = max_val
    train_env = train_env/max_val
    train_ask = train_ask/max_val

    return train_env, train_ask


native_period = 5  # Sampling rate for raw data
train_period = 30
train_length = 40  # Minutes back
minutes_back = int(train_period*train_length/60)
train_params = ['ETH-ask']
assets = 'ETH'

data_path = '/Users/mattsonthieme/Documents/Academic.nosync/Projects/RLTrading/data/BTC-ETH-XLM-CVC_5s_Copy.csv'

print("Creating {}_ask_p{}s_l{}m_env.pt".format(assets, train_period, minutes_back))

train_env, train_ask = parse_data(data_path, train_length, train_params, train_period, native_period)
torch.save(train_env, '{}_ask_p{}s_l{}m_env.pt'.format(assets, train_period, minutes_back))
torch.save(train_ask, '{}_ask_p{}s_l{}m_ask.pt'.format(assets, train_period, minutes_back))

















