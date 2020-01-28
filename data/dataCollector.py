import datetime
import time
import ccxt
import csv

exchange = ccxt.binance ()

eth = exchange.fetch_ticker('ETH/USDT')
del eth['info']
del eth['datetime']
del eth['timestamp']
write = ['time']

with open('highres.csv', 'w') as r:
    reader = csv.reader(r)
    try:
        row1 = next(reader)
    except:
        for i in eth:
            write.append(i)
        writer = csv.writer(r)
        writer.writerow(write)

print(write)


delay = 1

while True:

    write = []


    now = datetime.datetime.utcnow()
    eth = exchange.fetch_ticker('ETH/USDT')
    del eth['info']
    del eth['datetime']
    del eth['timestamp']

    write.append((now.hour*3600)+(now.minute*60)+now.second)
    for i in eth:
        write.append(eth[i])

    with open('highres.csv', "a") as f:
        writer = csv.writer(f)
        writer.writerow(write)


    time.sleep(delay)



