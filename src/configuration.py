
# Model parameters
model_type = 'multiphase'
# Learning parameters
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.001
EPS_DECAY = 30000  # Increasing in the hopes that it will help the model learn more about short term opportunities - takes ~80k steps to get to its lower bound
TARGET_UPDATE = 46000# 3000
POLICY_UPDATE = 3000  # Will update this actively in report (for now)
optimizer = optim.RMSprop(self.policy_net.parameters())
#optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  # Another optimizer - not sure which is optimal yet
total_steps = 0
BATCH_SIZE = 2048
hold_penalty = np.inf  # How long do we want to hold a falling asset?
max_reward_multiplier = 2
reward_turning_point = 160  # 40 mins at 15s period
reward_multiplier = 1  # Rewards are pretty sparse, we want to pump them up a bit


# Data parameters
data_folder = '../data/crypto/'
asset = 'ETH'
minutes_back = 20
period = 30
params = ['ask']





