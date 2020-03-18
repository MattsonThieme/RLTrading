# Standard strategy
#import configuration
import csv
import numpy as np

import matplotlib.pyplot as plt

#data_path = '../data/BTC-ETH-XLM-CVC_5s.csv'
data_path = 'ETH_1s_4.csv'
#data_path = 'VOO_copy.csv'

#assets = configuration.asset_names

'''
# 5s gnn collection
with open(data_path, 'r') as f:
    data = csv.reader(f)
    data = list(data)
    labels = data.pop(0)

    for row in data:
        for i, col in enumerate(row):
            prices[i].append(col)

data = np.array(prices, dtype=float)
'''

# Standard 1s collection
prices = [[]]
with open(data_path, 'r') as f:
    data = csv.reader(f)
    data = list(data)
    labels = data.pop(0)
    print(labels)

    for row in data:
        prices[0].append(row[5])

prices[0] = prices[0][:int(len(prices[0])*0.9)]

indices = [i for i in range(0, len(prices[0]), 1)]

length = len(prices[0])

prices = np.array(prices[0], dtype=float)
prices = prices[indices]


data = np.array([list(prices)], dtype=float)



def slope(last, current):
    return (current - last)

def profit(bought_price, sell_price, investment):
    fee = 0#0.00075

    buy_cost = investment*fee
    bought_shares = (investment - buy_cost)/bought_price
    sold_revenue = bought_shares*sell_price
    sold_cost = sold_revenue*fee
    profit = sold_revenue - sold_cost - buy_cost - investment


    #return sell_price*investment/bought_price - investment
    return profit


# Policy:

# If the last slope was zero or negative, and this slope is positive, buy
# If we've bought, and the slope goes to zero or negative, sell

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
        inv_track = []
        profit_track = []
        profit_trade = 0

        for i, ask in enumerate(asset):

            inv_track.append(investment)
            net.append(0)
            profit_track.append(profit_trade)

            curr_slope = slope(last_ask, ask)

            if bought == False:
                if (last_slope >= 0) and (curr_slope > 0):
                    bought = ask
                    sold = False
                    #net.append(0)

            if bought != False:
                hold_time += 1
                #net.append(0)

            if sold == False:
                if (last_slope >= 0) and (curr_slope < 0):
                    net[i] = profit(bought, ask, investment)
                    hold.append(hold_time)
                    investment += profit(bought, ask, investment)
                    profit_trade += profit(bought, ask, investment)
                    profit_track[i] = profit_trade
                    inv_track[i] = investment
                    hold_time = 0
                    bought = False
                    sold = None


            last_slope = curr_slope
            last_ask = ask

        num_trades = np.count_nonzero(np.array(net))

        print("\nAsset = {}, {} - {}, len: {}".format('VOO', asset[0], asset[len(asset)-1], len(asset)))
        print("Trades: ", num_trades)
        print("Avg hold: ", sum(hold)/len(hold))
        print("Avg profit:   $", sum(net)/len(net))
        print("Total profit: $", sum(net))


        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('exp', color=color)
        ax1.plot(asset, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
        ax2.plot(profit_track, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

iterate(data)




















