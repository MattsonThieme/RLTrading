
import ccxt
print(ccxt.exchanges)

exchange = ccxt.binance ()
markets = exchange.load_markets ()
#print(exchange.id, markets)

etheur1 = exchange.markets['ETH/USDT']

symbols = exchange.symbols

#print(exchange.id, symbols)

ethprice = exchange.fetch_ticker('ETH/USDT')['ask']

#print(ethprice)

import time

delay = 1
'''
for symbol in exchange.markets:
    print(symbol)
    print(exchange.fetch_order_book (symbol))
    time.sleep(delay)
'''

orderbook = exchange.fetch_order_book (exchange.symbols[0])
bid = orderbook['bids'][0][0] if len (orderbook['bids']) > 0 else None
ask = orderbook['asks'][0][0] if len (orderbook['asks']) > 0 else None
spread = (ask - bid) if (bid and ask) else None
#print (exchange.id, 'market price', { 'bid': bid, 'ask': ask, 'spread': spread })


print(exchange.fetch_ticker('ETH/USDT'))
print(exchange.rateLimit)