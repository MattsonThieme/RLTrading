# Standard strategy
import configuration
import csv
import numpy as np

import matplotlib.pyplot as plt

data_path = '../data/BTC-ETH-XLM-CVC_5s.csv'
#data_path = '../data/crypto/ETH/ETH_1s_4.csv'

assets = configuration.asset_names

prices = [[], [], [], []]

print(assets)

with open(data_path, 'r') as f:
    data = csv.reader(f)
    data = list(data)
    labels = data.pop(0)

    for row in data:
        for i, col in enumerate(row):
            prices[i].append(col)

data = np.array(prices, dtype=float)


def slope(last, current):
    return (current - last)

def would_profit(bought_price, current):
    raise NotImplementedError

def profit(bought_price, sell_price, investment):
    return sell_price*investment/bought_price - investment

# Policy:

# If the last slope was zero or negative, and this slope is positive, buy
# If we've bought, and the slope goes to zero or negative, sell

def iterate(data):
    sold = None
    bought = False
    asset = "BTC"
    asset = labels.index(asset+"-ask")
    last_slope = slope(data[asset][0], data[asset][1])
    last_ask = data[asset][0]
    investment = 1000  # No fees

    net = []
    hold = []

    for a, asset in enumerate(data):
        net = []
        hold = []
        last_slope = slope(data[a][0], data[a][1])
        last_ask = data[a][0]
        for i, ask in enumerate(asset):
            curr_slope = slope(last_ask, ask)

            if bought == False:
                if (last_slope <= 0) and (curr_slope > 0):
                    bought = ask
                    sold = False

            if bought == True:
                hold += 1

            if sold == False:
                if (last_slope >= 0) and (curr_slope < 0):
                    net.append(profit(bought, ask, investment))
                    bought = False
                    sold = None

            last_slope = curr_slope
            last_ask = ask

        print("\nAsset = ", assets[a])
        print("Trades: ", len(net))
        print("Avg profit:   $", sum(net)/len(net))
        print("Total profit: $", sum(net))

def iterate(data):


    net = []
    hold = []

    for a, asset in enumerate(data):
        sold = None
        bought = False
        investment = 1000  # No fees
        
        net = []
        hold_time = 0
        hold = []
        last_slope = slope(asset[0], asset[1])
        last_ask = asset[0]

        for i, ask in enumerate(asset):

            curr_slope = slope(last_ask, ask)

            if bought == False:
                if (last_slope <= 0) and (curr_slope > 0):
                    bought = ask
                    sold = False

            if bought != False:
                hold_time += 1

            if sold == False:
                if (last_slope >= 0) and (curr_slope < 0):
                    net.append(profit(bought, ask, investment))
                    hold.append(hold_time)
                    investment += profit(bought, ask, investment)
                    hold_time = 0
                    bought = False
                    sold = None


            last_slope = curr_slope
            last_ask = ask

        plt.plot(net)
        #plt.show()
        print("\nAsset = {}, {} - {}".format(assets[a], asset[0], asset[len(asset)-1]))
        print("Trades: ", len(net))
        print("Avg hold: ", sum(hold)/len(hold))
        print("Avg profit:   $", sum(net)/len(net))
        print("Total profit: $", sum(net))

iterate(data)




















