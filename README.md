# RLTrading

Deep reinforcement learning for algorithmic cryptocurrency trading.

## Summary

This repo contains all code necessary for training a deep Q-learning agent to trade cryptocurrency, including scripts for:

1. Environment configuration
2. Data acquisition via [CCXT](https://github.com/ccxt/ccxt)
3. Pre-processing
4. Training/deployment

Sections 3 and 4 are integrated and controlled via a [configuration script](src/configuration.py). See details in the following sections.


Before we move forward, not that that the current implementation is not restricted to trading cryptocurrencies, and can be equally applied to regular stocks trading on any public exchanges. While I haven't integrated the functionality just yet (should be completed soon) I've included 20 years of DJIA data [here](data/djia/) in case you would like to start experimenting on your own. 


## Setup

Note: These instructions and setup scripts assume execution in Linux or MacOS environments with conda (for conda installation, see their [install page](https://docs.conda.io/en/latest/miniconda.html)), and as such may require some tweaking for full functionality on Windows.

Step one: clone this repo:

```{shell}
git clone https://github.com/MattsonThieme/RLTrading.git
cd RLTrading
```

Then run the following setup script to build and configure a new conda environment:

```{shell}
bash env_setup.sh
```

This will create a new environment named `rltrade`. To start working, use conda to enter the environment.

```{shell}
conda activate rltrade
```

## Data Acquisition

For this project, I have included sample data so you can get started right away, but if you would like to collect your own data on an asset I haven't included, or at a frequency I haven't sampled, or just a larget dataset (GitHub limits file sizes to 100Mb) a script is available in [data/dataCollector.py](data/dataCollector.py). 

To run the collection script, I recommend using a terminal multiplexer like [tmux](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/) on a small AWS instance. Edit parameters at the top of the file including 'symbol' and 'delay' then run:

```{shell}
python dataCollector.py
```

This will create and continuously append file with your target asset and sampling period (delay). We'll see in subsequent sections how to use this new file for training (it's as simple as setting a single variable in the configuration file).

## Training

The policy network is a custom network which attends over price movements and environmental parameters separately. Details can be found in the 
