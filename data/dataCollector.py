
# Note: this collection script works but is a bit clunky - I'll push
# a better version in the near future.

# Also note: can't query most exchanges more than 5x/second or they'll ban 
# you temporarily - it's mostly just annoying, so make sure you don't query too frequently.

import datetime
import time
import ccxt
import csv

exchange = ccxt.binance ()

asset = 'ETH'
delay = 1

symbol = '{}/USDT'.format(asset)

eth = exchange.fetch_ticker(symbol)

to_delete = ['average', 'percentage', 'change', 'previousClose', 'info',
             'datetime', 'timestamp', 'symbol', 'baseVolume', 'quoteVolume']

for i in to_delete:
    del eth[i]

write = []

with open('{}_{}s.csv'.format(asset, str(delay)), 'w') as r:
    reader = csv.reader(r)
    try:
        row1 = next(reader)
    except:
        for i in eth:
            write.append(i)
        writer = csv.writer(r)
        writer.writerow(write)


while True:

    try:
        eth = exchange.fetch_ticker(symbol)
        start = time.time()

        write = []
        
        for i in to_delete:
            del eth[i]

        for i in eth:
            write.append(eth[i])

        with open('{}_{}s.csv'.format(asset, str(delay)), "a") as f:
            writer = csv.writer(f)
            writer.writerow(write)

        end = time.time()

        time.sleep(delay - (end - start))

    except:
        # Occasionally the exchange won't respond, this lets it catch itself and continue
        time.sleep(delay)



