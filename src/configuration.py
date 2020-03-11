# Model parameters
model_type = 'multiphase'  # Current custom network
mem_capacity = 1000        # Size of the memory replay buffer

# Learning parameters
GAMMA = 0.999              # Scaling significance of reward from the next step
EPS_START = 0.9            # This model uses an epsilon greedy strategy, this sets the initial value of epsilon
EPS_END = 0.001            # Lower bound of epsilon
EPS_DECAY = 10000          # Steps between upper and lower bounds on epsilon
TARGET_UPDATE = 900        # Examples between target network updates (want to keep this ~10x policy_update)
POLICY_UPDATE = 90         # Examples between policy updates
BATCH_SIZE = 2048          # Batch size

# Data/Env parameters
data_path = '../data/crypto/ETC_1s.csv'
asset = 'ETC'              # Target asset
minutes_back = 10          # Assumes a 1s sampling rate in raw data, how far to look into the past
period = 15                # Assumes a 1s sampling rate in raw data, how many seconds between market queries
params = ['ask']           # Parameters to consider - currently, we are only setup to look at ask price
report_freq = 10           # Number of profitable trades to accumulate before reporting

# Financial parameters
fee = 0.0                  # 0.00075 for Binance - 0% currently because Robinhood is on the scene!
investment_scale = 1000    # How much of the asset we purchase with each trade. Value of 1000 = $1,000 invested with each trade
