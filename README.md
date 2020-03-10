# RLTrading

Deep reinforcement learning for algorithmic cryptocurrency trading.

## Summary

This repo contains all code necessary for training a deep Q-learning agent to trade cryptocurrency, and includes scripts for:

1. Environment configuration
2. Data acquisition via [ccxt](https://github.com/ccxt/ccxt)
3. Pre-processing
4. Training/deployment

Sections 3 and 4 are integrated and controlled via a [configuration script](src/configuration.py). See details in the following sections.

Before we move forward, note that that the current implementation is not restricted to trading cryptocurrencies, and can be applied equally to any stocks trading on any public exchanges. While I haven't integrated the functionality just yet (should be completed soon) I've included 20 years of DJIA data [here](data/djia/) in case you would like to start experimenting on your own. 

## Setup

Note: These instructions and setup scripts assume execution in Linux or MacOS environments with conda (for conda installation, see their [install page](https://docs.conda.io/en/latest/miniconda.html)), and as such may require some tweaking for full functionality on Windows.

Step one: clone this repo:

```{shell}
$ git clone https://github.com/MattsonThieme/RLTrading.git
$ cd RLTrading
```

Then run the following setup script to build and configure a new conda environment:

```{shell}
$ bash env_setup.sh
```

This will create a new environment named `rltrade`. Activate the conda environment, and we're ready to begin working.

```{shell}
$ conda activate rltrade
```

## Data Acquisition

For this project, I have included sample data so you can get started right away, but if you would like to collect your own data on an asset I haven't included, or at a frequency I haven't sampled, or just a larget dataset (GitHub limits file sizes to 100Mb) a script is available in [data/dataCollector.py](data/dataCollector.py). 

To run the collection script, I recommend using a terminal multiplexer like [tmux](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/) on a small AWS instance. Edit parameters at the top of the file including 'symbol' and 'delay' then run:

```{shell}
python dataCollector.py
```

This will create and continuously append a file with your target asset's ask price once every sampling period (delay). We'll see in subsequent sections how to use this new file for training (it's as simple as setting a single variable in the configuration file).

## Training

Change directory into `src/`:

```{shell}
$ cd src
```

This folder contains only two files: `configuration.py` and `dqn.py`. Within the `configuration.py` file, you will find all the most important training parameters. 

The most important training parameters are contained in `configuration.py`. Here, you will be able to edit parameters like the size of the replay buffer, update frequencies, and batch size. However, we don't need to edit anything to get started. All necessary data is already contained in [data/](../data/), so we will get right into training.

To initiate training, ensure that you are in the `(rltrade)` environment, and run:

```{shell}
$ python dqn.py
```

When training has begun, a report will be printed after every 20 profitable trades (this frequency can be modified in `configuration.py`). The report will look like the following:

```{shell}
Global start: $12.43, current: :$12.87  -- (995/997)
Market moved $-0.16 over the session
Start: $12.434, current: $12.876
     Session wins: 20 @ $ 0.98, avg hold: 2.0 steps
     Session loss: 13 @ $-0.51, avg hold: 2.0 steps
     Session Net:  $8.45
     Episode total: $33.84
``` 

Here, we are shown information about the global start (first ask price in the dataset), the as price of the current step, as well as how much the market moved over the last 20 profitable trades. The term `avg hold` represents how long, on average, we held the asset before selling it. We can also see in this example, that we've made a total of $33.84 over the entire episode, and $8.45 over the last 20 wins and (in this case) 13 losses.

```diff
- Disclaimer: the provided sample dataset is relatively small. Before deploying any trained models and making real trades, you will probably want to collect more data and validate the model over a longer time period. 
```
## Model



## I/O


The policy network is a custom network which attends over price movements and environmental parameters separately. Details can be found in the 
