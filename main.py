# from binance.client import Client
import numpy as np
import pandas as pd
import smtplib
import time
# import yaml
import os
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import datetime

import btalib

# import tulipy

import talib
from talib import RSI, BBANDS, SMA, MOM
import pandas_ta as ta

from ta import add_all_ta_features
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.utils import dropna

from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = [12, 7]

plt.rc('font', size=14)


from decimal import Decimal, getcontext

# from binance.websockets import BinanceSocketManager
# from twisted.internet import reactor

LOG_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

binance = "https://api.binance.com"
avg_price = "/api/v3/avgPrice"
all_prices = "/api/v3/ticker/price"

# CONFIG = yaml.load(open('./CONFIG.yml'))

# API_KEY = CONFIG['binance_api']['key']
# API_SECRET = CONFIG['binance_api']['secret']


API_KEY = os.environ.get('binance_api')
API_SECRET = os.environ.get('binance_secret')

# client = Client(API_KEY, API_SECRET)

# against ETH
SYMBOLS = ('ADA', 'ADX', 'BAT', 'BCC', 'DASH', 'EOS', 'IOTA',
        'LTC', 'NEO', 'OMG', 'STORJ', 'XLM', 'NANO', 'XRP', 'XVG', 'ZEC')

RSI_N = 14
RSI_THRESHOLD = 8
RUN_INTERVAL_MINS = 10

coin_price = {'error': False}



def coin_trade_history(msg):
    ''' define how to process incoming WebSocket messages '''
    if msg['e'] != 'error':
        print(msg['c'])
        coin_price['last'] = msg['c']
        coin_price['bid'] = msg['b']
        coin_price['last'] = msg['a']
    else:
        coin_price['error'] = True



def pure_python_rsi(prices, n):
    prices = prices.astype(np.float64)
    deltas = prices.diff()

    seed = deltas[:n]

    smooth_up = seed[seed > 0].sum()/(n - 1)
    smooth_down = -seed[seed < 0].sum()/(n - 1)

    rsi = np.zeros_like(prices)
    rsi[:n] = np.nan

    for i in range(n, len(prices)):
        delta = deltas[i]

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        smooth_up = (upval - smooth_up) / n + smooth_up
        smooth_down = (downval - smooth_down) / n + smooth_down
        rs = smooth_up / smooth_down
        rsi[i] = 100. - 100. / (1. + rs)

    return pd.Series(rsi)


def RSI2(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    try:
        u[u.index[period-1]] = np.mean(u[:period]) #first value is sum of avg gains
        u = u.drop(u.index[:(period-1)])
        d[d.index[period-1]] = np.mean(d[:period]) #first value is sum of avg losses
        d = d.drop(d.index[:(period-1)])

        rs = u.ewm(com=period - 1, adjust=False).mean() / d.ewm(com=period - 1, adjust=False).mean()
        rs = 100 - 100 / (1 + rs)
    except IndexError:
        rs = -1
    # rs = pd.stats.moments.ewma(u, com=period-1, adjust=False) / \
    #      pd.stats.moments.ewma(d, com=period-1, adjust=False)

    return rs


def isNaN(num):
    return num != num


def get_api_klines(pair, interval="1d"):

    url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval={interval}" \
          % {'pair': pair, 'interval': interval}

    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    response = session.get(url)

    try:
        klines = response.json()
    except requests.exceptions.ConnectionError:
        tslog("Connection error")
        klines = {}

    return klines


def tslog(msg="", logfilesufix="_log.txt", path=""):

    global LOG_FILE_PATH
    global start

    if LOG_FILE_PATH:
        path = LOG_FILE_PATH

    dateTimeObj = datetime.datetime.now()

    timestampStr = dateTimeObj.strftime("%Y-%m-%d")

    if path:
        timestampStr = path + "//" + timestampStr

    f = open(timestampStr + logfilesufix, "a")

    partial = round((time.time() - start), 1)

    f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z ") + ";" + str(partial) + ";" + msg + "\n")
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z ") + " - " + str(partial) + " : " + msg)


def bbp(price):
    # bolinger bands %
    multiplier = 1000000

    up, mid, low = BBANDS(price['Close'].values*multiplier, timeperiod=5, nbdevup=2, nbdevdn=2, matype=talib.MA_Type.T3)
    # multiply to avoid very small numbers leading to inf

    # up, mid, low = up * mutiplier, mid * mutiplier, low * mutiplier

    bbp = (((price['Close']*multiplier) - low) / (up - low))

    if isNaN(bbp.values[-1]):
        print("ok")
    return bbp


def bbww(price):
    multiplier = 1000000
    # Bollinger Band Width = (Upper Band - Lower Band) / Simple Moving Average for the same period.
    up, mid, low = BBANDS(price['Close'].values * multiplier, timeperiod=5, nbdevup=2, nbdevdn=2, matype=talib.MA_Type.T3)

    # up, mid, low = up * 10000000, mid * 10000000, low * 10000000

    sma = SMA(price['Close'].values, timeperiod=14) * multiplier

    bbw = (up - low) / (sma)

    return bbw


def isSupport(df,i):
    support = df['Low'][i] < df['Low'][i-1]  and df['Low'][i] < df['Low'][i+1] and df['Low'][i+1] < df['Low'][i+2] and df['Low'][i-1] < df['Low'][i-2]
    return support

def isResistance(df,i):
    resistance = df['High'][i] > df['High'][i-1]  and df['High'][i] > df['High'][i+1] and df['High'][i+1] > df['High'][i+2] and df['High'][i-1] > df['High'][i-2]
    return resistance

def plot_all():

    def isFarFromLevel(l):
        return np.sum([abs(l - x) < s for x in levels]) == 0

    levels = []

    for i in range(2, df.shape[0] - 2):
        if isSupport(df, i):
            l = df['Low'][i]
            if isFarFromLevel(l):
                levels.append((i, l))
        elif isResistance(df, i):
            l = df['High'][i]
            if isFarFromLevel(l):
                levels.append((i, l))

    fig, ax = plt.subplots()

    candlestick_ohlc(ax,df.values,width=0.6, \
                   colorup='green', colordown='red', alpha=0.8)

    date_format = mpl_dates.DateFormatter('%d %b %Y')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    fig.tight_layout()

    for level in levels:
        plt.hlines(level[1],xmin=df['Date'][level[0]],\
               xmax=max(df['Date']),colors='blue')
    fig.show()


if __name__ == '__main__':

    # bsm = BinanceSocketManager(client)

    # exchange_info = client.get_all_tickers()

    # url = "https://api.binance.com/api/v3/ticker/price"

    pd.set_option('display.precision', 12)

    url = "https://api.binance.com/api/v3/ticker/24hr"

    # trading_view()
    start = time.time()

    response = requests.get(url)

    all_tickers = response.json()

    df_tickers = pd.DataFrame(all_tickers,
                              columns=['symbol', 'priceChange',
                                       'priceChangePercent', 'weightedAvgPrice', 'prevClosePrice',
                                       'lastPrice', 'lastQty', 'bidPrice', 'askPrice', 'openPrice', 'highPrice',
                                       'lowPrice', 'volume', 'quoteVolume', 'openTime', 'closeTime', 'firstId', 'lastId',
                                       'counts'])

    df_tickers = df_tickers.set_index('closeTime')

    df_tickers['priceChange'] = df_tickers.priceChange.astype(float)
    df_tickers['priceChangePercent'] = df_tickers.priceChangePercent.astype(float)
    df_tickers['weightedAvgPrice'] = df_tickers.weightedAvgPrice.astype(float)
    df_tickers['prevClosePrice'] = df_tickers.prevClosePrice.astype(float)
    df_tickers['lastPrice'] = df_tickers.lastPrice.astype(float)
    df_tickers['lastQty'] = df_tickers.lastQty.astype(float)
    df_tickers['bidPrice'] = df_tickers.bidPrice.astype(float)
    df_tickers['askPrice'] = df_tickers.askPrice.astype(float)
    df_tickers['openPrice'] = df_tickers.openPrice.astype(float)
    df_tickers['highPrice'] = df_tickers.highPrice.astype(float)
    df_tickers['lowPrice'] = df_tickers.lowPrice.astype(float)
    df_tickers['volume'] = df_tickers.volume.astype(float)
    df_tickers['quoteVolume'] = df_tickers.quoteVolume.astype(float)
    df_tickers['openTime'] = df_tickers.openTime.astype(float)
    # df_tickers['closeTime'] = df_tickers.closeTime.astype(float)
    df_tickers['firstId'] = df_tickers.firstId.astype(float)
    df_tickers['lastId'] = df_tickers.lastId.astype(float)
    df_tickers['counts'] = df_tickers.counts.astype(float)

    df_btc = df_tickers[df_tickers['symbol'].str.endswith('BTC')]
    df_usdt = df_tickers[df_tickers['symbol'].str.endswith('USDT')]

    df_usdt = df_usdt.sort_values('quoteVolume', ascending=False).head(30)
    df_btc = df_btc.sort_values('quoteVolume', ascending=False).head(30)

    # df_final = pd.concat([df_usdt, df_btc], ignore_index=True, axis=0)

    df_final = pd.concat([df_usdt, df_btc], axis=0)

    coins_tracked = []
    rsi_values = []

    for index, coin in df_final.iterrows():

        if coin['symbol'][-3:] == 'BTC' or coin['symbol'][-4:] == 'USDT':

            coins_tracked.append(coin['symbol'])

    while True:

        total = len(coins_tracked)

        df_final = df = pd.DataFrame(columns=['pair', 'rsi_1h', 'rsi_4h', 'rsi_1d', 'rsi_1w',
                                              'bb_bbp_1h', 'bb_bbp_4h', 'bb_bbp_1d', 'bb_bbp_1w',
                                              'bb_bbw_1h', 'bb_bbw_4h', 'bb_bbw_1d', 'bb_bbw_1w'])
        df_rsi = pd.DataFrame()

        for n, pair in enumerate(coins_tracked):

            # valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
            tslog("Computing pair: " + pair + " - " + str(n+1) + "/" + str(total))

            tfs = ['1h', '4h', '1d', '1w']

            period = 30

            rsi_set = []

            bbp_set = []

            for i, tf in enumerate(tfs):

                # get timestamp of earliest date data is available
                # timestamp = client._get_earliest_valid_timestamp(pair, tf)

                if tf =='1w':
                    period = 300
                elif tf == '1d':
                    period = 100

                # bars = client.get_klines(symbol=pair, interval=tf)

                # request historical candle (or klines) data
                # bars = client.get_historical_klines(pair, tf, '{} days ago UTC'.format((period + 3) // 2))

                # bars = client.get_historical_klines(pair, tf, timestamp, 1000)
                bars = get_api_klines(pair, tf)

                df_bars = pd.DataFrame(bars, columns=['Open time', 'Open', 'High', 'Low', 'Close',
                                                      'Volume', 'Close time', 'Quote asset volume',
                                                      'Number of trades', 'Taker buy base asset volume',
                                                      'Taker buy quote asset volume', 'Ignore'])

                df_bars = dropna(df_bars)
                df_bars['Open'] = pd.to_numeric(df_bars['Open'], errors='coerce')
                df_bars['Close'] = pd.to_numeric(df_bars['Close'], errors='coerce')
                df_bars['High'] = pd.to_numeric(df_bars['High'], errors='coerce')
                df_bars['Low'] = pd.to_numeric(df_bars['Low'], errors='coerce')
                df_bars['Quote asset volume'] = pd.to_numeric(df_bars['Quote asset volume'], errors='coerce')

                # Initialize Bollinger Bands Indicator
                indicator_bb = BollingerBands(close=df_bars["Close"], window=20, window_dev=2)

                # Add Bollinger Bands features
                # df_bars['bb_bbm'] = indicator_bb.bollinger_mavg()
                # df_bars['bb_bbh'] = indicator_bb.bollinger_hband()
                #df_bars['bb_bbl'] = indicator_bb.bollinger_lband()

                # Add Bollinger Band high indicator
                # df_bars['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()

                # Add Bollinger Band low indicator
                # df_bars['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

                # Add Width Size Bollinger Bands
                df_bars['bb_bbw_' + tf] = indicator_bb.bollinger_wband()

                # Add Percentage Bollinger Bands
                df_bars['bb_bbp_' + tf] = indicator_bb.bollinger_pband()


                # df_bars.insert(loc=0, column='pair', value=pair)
                # df_bars.insert(loc=1, column='tf', value=tf)
                # df_bars['pair'] = pair
                # df_bars['High'] = df_bars.High.astype(np.float64)
                # df_bars['Low'] = df_bars.Low.astype(np.float64)
                # df_bars['Close'] = df_bars.Close.astype(np.float64)

                df_bars['rsi_' + tf] = RSIIndicator(close=df_bars["Close"]).rsi()

                # df_bars = add_all_ta_features(
                #    df_bars, open="Open", high="High", low="Low", close="Close", volume="Quote asset volume")

                # print(df_bars.info())
                # use Python Function

                # rsi = pure_python_rsi(df_bars['Close'], 14)

                # df_bars['rsi'] = rsi

                # df_bars['bbp_val'] = bbp(df_bars).values

                # bbw = bbww(df_bars)

                # momentum of the close prices
                # df_bars['mom'] = MOM(df_bars['Close'], timeperiod=5)

                # df_bars.ta.log_return(cumulative=True, append=True)
                # df_bars.ta.percent_return(cumulative=True, append=True)

                # Create your own Custom Strategy

                # df_bars = df_bars.set_index('Open time')

                # " closings = np.asarray(bars, dtype=np.float)[-RSI_N - 1:, 4]

                # use TA-Lib
                # rsi = RSI(df_bars['Close'], timeperiod=14)

                # if len(df_bars) < 15 and len(df_bars) > 0:
                #     print("Length < 14...")
                #     rsi = RSI(df_bars['Close'], len(df_bars)-1)
                # elif len(df_bars) > 13:
                #     rsi = RSI(df_bars['Close'], 14)
                # else:
                #     rsi_set.append(-1)
                #     continue
                print(tf + ": " + str(df_bars.tail(1)['rsi_' + tf].values.max()))

                df_rsi['rsi_' + tf] = df_bars.tail(1)['rsi_' + tf].values
                df_rsi['bb_bbp_' + tf] = df_bars.tail(1)['bb_bbp_' + tf].values
                df_rsi['bb_bbw_' + tf] = df_bars.tail(1)['bb_bbw_' + tf].values

                # rsi_set.append(list(df_bars.tail(1)[['RSI_14', 'BBL', 'BBM', 'BBU']]))

                # df_final.loc[i] = list(df_bars.tail(1)[['pair', 'RSI_14', 'BBL', 'BBM', 'BBU']])

                # df_final = pd.concat([df_final, df_bars.tail(1)], axis=1)
            df_rsi['pair'] = pair

            df_final = pd.concat([df_final, df_rsi], axis=0)

            # rsi_values.append((pair, rsi_set, bbp_set))

        # print('\n'.join('{0:>8} {1:.2f}'.format(symbol, rsi) for (symbol, rsi) in rsi_values))
        # rsi_values = list(filter(lambda x: x[1] < RSI_THRESHOLD, rsi_values))

        # df = pd.DataFrame(rsi_values, columns=['Pair', 'rsi', 'bbp'])

        # df[['1h', '4h', '1d', '1w']] = pd.DataFrame(df.rsi.tolist(), index=df.index)

        # df[['bbp_1h', 'bbp_4h', 'bbp_1d', 'bbp_1w']] = pd.DataFrame(df.bbp.tolist(), index=df.index)

        df_buy = df_final[(df_final[['rsi_1h', 'rsi_4h', 'rsi_1d', 'rsi_1w']] < 40).all(axis=1)]

        if len(df_buy.index) == 0:

            df_buy = df_final[(df_final[['rsi_1h', 'rsi_4h', 'rsi_1d']] < 40).all(axis=1)]

        df_sell = df_final[(df_final[['rsi_1h', 'rsi_4h', 'rsi_1d', 'rsi_1w']] > 80).all(axis=1)]

        if len(df_sell.index) == 0:
            df_sell = df_final[(df_final[['rsi_1h', 'rsi_4h', 'rsi_1d']] > 80).all(axis=1)]


        # df_buy = df_final[(df_final['1h'] < 40) & (df_final['4h'] < 40) & (df_final['1d'] < 40) & (df_final['1w'] < 40)]

        # df_sell = df[(df['1h'] > 70) & (df['4h'] > 70) & (df['1d'] > 70) & (df['1w'] > 70)]

        end = time.time()

        print(end-start)

        time.sleep(60 * RUN_INTERVAL_MINS)


