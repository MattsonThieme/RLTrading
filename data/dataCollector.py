import datetime
import time
import ccxt
import csv

exchange = ccxt.binance ()

symbol = 'ETH/USDT'

eth = exchange.fetch_ticker(symbol)

to_delete = ['average', 'percentage', 'change', 'previousClose', 'info',
             'datetime', 'timestamp', 'symbol', 'baseVolume', 'quoteVolume']

for i in to_delete:
    del eth[i]

write = ['time']

with open('ETH_1s.csv', 'w') as r:
    reader = csv.reader(r)
    try:
        row1 = next(reader)
    except:
        for i in eth:
            write.append(i)
        writer = csv.writer(r)
        writer.writerow(write)

delay = 1

while True:

    write = []

    now = datetime.datetime.utcnow()
    eth = exchange.fetch_ticker(symbol)
    for i in to_delete:
        del eth[i]

    write.append((now.hour*3600)+(now.minute*60)+now.second)
    for i in eth:
        write.append(eth[i])

    with open('ETH_1s.csv', "a") as f:
        writer = csv.writer(f)
        writer.writerow(write)


    time.sleep(delay)



