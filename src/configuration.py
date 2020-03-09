# Model parameters
model_type = 'multiphase'
mem_capacity = 100000

# Learning parameters
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.001
EPS_DECAY = 30000  # Increasing in the hopes that it will help the model learn more about short term opportunities - takes ~80k steps to get to its lower bound
TARGET_UPDATE = 46000
POLICY_UPDATE = 3000  # Will update this actively in report (for now)
total_steps = 0
BATCH_SIZE = 2048

# Data/Env parameters
data_path = '../data/crypto/ETH/ETH_1s_4.csv'
asset = 'ETH'
minutes_back = 20  # Assumes a 1s sampling rate
period = 30
params = ['ask']

# Financial parameters
fee = 0.0  # 0.00075 for Binance - 0% currently because Robinhood is on the scene!
investment_scale = 1000  # How much of the asset we purchase with each trade
