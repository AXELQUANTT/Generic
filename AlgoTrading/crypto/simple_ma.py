"""
Small code sample to try different features 
of Backtrader and cryptodatadownload. This serves
as an example to start understanding the logic
of how to use backtrader
"""

import pandas as pd
import backtrader as bt
import numpy as np



def load_data(coin:str, exchange:str):
    """
    Function will load data for the data found at CryptoDataDownload.com
    """
    sym = coin.split("/")[0]    # get the symbol in front of the slash, ie BTC from BTC/USD
    sym_base = coin.split("/")[1]   # get the base symbol, ie USD in BTC/USD
    URL = f"https://www.cryptodatadownload.com/cdd/{exchange}_{sym+sym_base}_d.csv"   # path to file
    data = pd.read_csv(URL, skiprows=1)    # use pandas to read file from site
    data['Date'] = pd.to_datetime(data['Date'])
    data['datetime'] = data['Date'].apply(lambda x: int(x.timestamp() * 1000))
    data["datetime"] = data["datetime"].values.astype(dtype='datetime64[ms]')
    data.sort_values('datetime', ascending=True, inplace=True)  # we need to sort the data to have most recent data last in file
    data.set_index('datetime', inplace=True)
    data.drop(columns=[f'Volume {sym}', f'Volume {sym_base}', 'tradecount', 'Symbol', 'Unix', 'Date'], inplace=True) # drop columns we dont need
    return data  # return our large and combined dataframe



#class pt_spread(bt.Indicator):
#    lines = ('pt_spread',)
#    params = (('value',5),)

#    def __init__(self):
#        self.lines.pt_spread = 20.0 - self.data1.close - self.data0.close

class PairTrading(bt.Strategy):
    # This is where we define the main parts of our PairTrading strategy
    params = {'hw':10}

    def __init__(self):
        # Initialize our signal
        self.signal = np.log(self.datas[0].close/self.datas[1].close) - 

    def next(self):
        # Next function defines the logic of what to do
        # once we get to the next time window
        #print('aaa')
        None



btc = 'BTC/USDT'
eth = 'ETH/USDT'
btc_binance = load_data(btc,'Binance')
eth_binance = load_data(eth,'Binance')

# We first need to specify the data model that will be used
btc_binance = bt.feeds.PandasData(dataname=btc_binance,
                           open='Open',
                            high='High',
                            low='Low',
                            close='Close',
                            volume=None,
                            openinterest=None)

eth_binance = bt.feeds.PandasData(dataname=eth_binance,
                           open='Open',
                            high='High',
                            low='Low',
                            close='Close',
                            volume=None,
                            openinterest=None)

# Create a cerebro engine instance for backtrader
cerebro = bt.Cerebro()

# Add both datafeeds to the engine
cerebro.adddata(btc_binance, name='btc')
cerebro.adddata(eth_binance, name='eth')
cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='areturn')
cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=3.95)
cerebro.addanalyzer(bt.analyzers.Returns)
cerebro.addanalyzer(bt.analyzers.DrawDown)
cerebro.addstrategy(PairTrading)

results = cerebro.run()
# Is there any way to ignore the volume?
cerebro.plot(volume=False)