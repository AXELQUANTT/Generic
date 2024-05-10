# Define set of imports needed for the script
import pandas as pd
import numpy as np
import argparse
import datetime
from typing import List, Any
import matplotlib.pyplot as plt
import mplfinance as mpf
import scikits.timeseries as ts
#import scikits.timeseries as ts

# Create argparser to read the file from the command line
#parser = argparse.ArgumentParser(
#                    prog='Futures_PairTrading',
#                    description='Program generates a pair trading strategy with post-trade analysis',
#                    epilog='Enjoy the analysis')

#parser.add_argument('--input_data',
#                    help='Path containing the CSV with the futures data',
#                    type=str,
#                    default='/home/axelbm23/Code/webb_traders_assignement/ESNQ')
                    #required=True)
#args = parser.parse_args()


def create_settings(ewma_hl,min_spread,max_spread,check_ba_spread=False,ba_spread=0.0,check_volume=False) -> dict[dict[str,Any]]:

    signal_settings : dict = {'ewma_halflife':ewma_hl,#this ewma halflife is in minutes!!!
                            'ewma_adjust':False,# This bool indicates if the weights of the moving average should be corrected or not
                            'ewma_min_periods':ewma_hl*2}

    order_settings : dict = {'ba_spread':ba_spread,
                            'min_spread':min_spread,#between 0 and this value, Optimal_investment is zero (1.0)
                            'max_spread':max_spread,#at this value, investment should be maxixum (2.0) # TO-DO: In order to determine those min and max spread, take a distribution of the signal and compute quantiles. Max spread should be around quantile 0.9, min_spread around 0.75
                            'step_spread':abs(max_spread-min_spread)/2,
                            'check_volume':check_volume,# If True, order_volume < quote_volume, else volume executed will be restricted to quote_volume
                            'check_ba_spread':check_ba_spread
                            }
    
    investment_settings : dict = {'cash':1_000_000,
                                'inv_per_trade':250_000,
                                'max_inv':750_000,
                                'min_inv_factor':0.15# factor used to determine whether rebalance should take place
                                }

    settings : dict[dict[str,Any]] ={
        'signal_sett':signal_settings,
        'inv_sett':investment_settings,
        'order_sett':order_settings
    }

    return settings

def import_data(path:str, sett:dict[str,float]) -> list[pd.DataFrame]:
    # TO-DO: This needs to be replaced with the argparse argument
    df = pd.read_csv(path,
                     dtype={'symbol':'string',
                            'year':'int64',
                            'month':'int64',
                            'day':'int64',
                            'hour':'int64',
                            'minute':'int64',
                            'second':'int64',
                            'bid_price':'float64',
                            'bid_size':'int64',
                            'ask_price':'float64',
                            'ask_size':'int64',
                            'mid_price':'float64'})
    
    # Upon preliminary exploration,month,day,hour,minute and second column
    # can be reduced into a single one with a datetime type. Six columns
    # will be merged to one.
    df['date'] = pd.to_datetime(df[['year','month','day','hour','minute','second']])
    df.drop(['year','month','day','hour','minute','second'],inplace=True,axis=1)
    
    # Sort the array by date and reset index
    df.sort_values(['date','symbol'],inplace=True)
    df.reset_index(inplace=True)

    # Pivot the dataframe to make analysis easier
    df = df.pivot(index='date',columns='symbol',values=['bid_price','bid_size','ask_price','ask_size','mid_price'])

    # Check also if there are any NaNs on the dataset we need to deal with
    # (there isn't any)
    if df.isnull().values.any():
        print('Dataframe contains some Nans!')


    # Split the dataset between training, cross validation 
    # and testing dataframes. Considering training will
    # only be used to compute the beta of one price wrt to
    # the other, it does not need to be that big.
    train_date = df.index[int(df.shape[0]*sett['train_split'])+1]
    cv_date = df.index[int(df.shape[0]*sett['crossval_split'])+1]


    train_df = df.loc[df.index < train_date,:]
    cv_df = df.loc[((df.index>=train_date)&(df.index<cv_date)),:]
    test_df = df.loc[df.index>=cv_date,:]

    # test 0. train_df, cv_df and test_df should have no overlaps
    overlap = set(train_df.index) & set(cv_df.index) & set(test_df.index)
    if len(overlap)!=0:
        raise ValueError('Split is not correctly perform, check')
    
    return [train_df,cv_df,test_df]

def compute_dev_width(signal:pd.Series,pctiles:list[float]) -> list[float]:
    # Return an array with the computed quantils for the input percentiles
    computed_qtiles = []

    for pcti in pctiles:
        # since we'll have nans according to the ewma min_period, use nanquantile (ignores nans)
        # also the values retrieved will be in the bps scale, so round them to make
        # the entry,exit points a bit more stable
        computed_qtiles.append(round(np.nanquantile(signal,pcti),5))
    
    return computed_qtiles

def compute_signal(df:pd.DataFrame, y_ticker, x_ticker, ewma_hl,min_periods,adjust_ewma=False) -> pd.Series:
    ratio = np.log(df['mid_price'][y_ticker]/df['mid_price'][x_ticker])
    ewma = ratio.ewm(halflife=ewma_hl,adjust=adjust_ewma,min_periods=min_periods).mean()
    signal = ratio-ewma

    return signal

class BackTester():
    def __init__(self, settings:dict[dict[str,Any]], data:pd.DataFrame):
        # Set some state variables used for the simulation
        self.data = data
        self.ticker_y = 'ESc1'
        self.ticker_x = 'NQc1'
        self.pos_y = 0.0
        self.pos_x = 0.0
        self.money_y = 0.0
        self.money_x = 0.0
        self.price_y = np.nan
        self.price_x = np.nan
        self.ref_price_y = np.nan
        self.ref_price_x = np.nan
        self.opt_inv = 0.0 # optimal_investment
        self.curr_inv = 0.0 # current_investment
        #self.pair_dev = 0.0
        self.trades = []
        self.accounts = []
        
        # Investment Settings
        self.cash = settings['inv_sett']['cash']
        self.inv_per_trade = settings['inv_sett']['inv_per_trade']
        self.max_inv = settings['inv_sett']['max_inv']
        self.min_inv_factor = settings['inv_sett']['min_inv_factor']

        # Signal settings
        self.ewma_halflife = settings['signal_sett']['ewma_halflife']
        self.ewma_adjust = settings['signal_sett']['ewma_adjust']
        self.ewma_min_periods = settings['signal_sett']['ewma_min_periods']
        
        # Order settings
        self.ba_spread = settings['order_sett']['ba_spread']
        self.min_spread = settings['order_sett']['min_spread']
        self.max_spread = settings['order_sett']['max_spread']
        self.step_spread = settings['order_sett']['step_spread']
        
        
    def compute_signal(self):
        ratio = np.log(self.data['mid_price'][self.ticker_y]/self.data['mid_price'][self.ticker_x])
        ewma = ratio.ewm(halflife=self.ewma_halflife,adjust=self.ewma_adjust,min_periods=self.ewma_min_periods).mean()
        self.data['signal'] = ratio-ewma

    def _compute_opt_investment(self,signal:float) -> float:
        
        # In case signal is not available, then return nan
        if signal==np.nan:
            return np.nan

        # If spread is lower than min_spread, opt_inv is 0.
        if abs(signal) < self.min_spread:
            return 0.0
        
        else:
            # basically means abs(signal) >= self.min_spread:
            multiplier = (abs(signal)- self.min_spread)//self.step_spread + 1.0
        
        if signal>0.0:
            optimal_investment = -1.0*multiplier*self.inv_per_trade
        else:
            optimal_investment = multiplier*self.inv_per_trade
        
        return optimal_investment
    
    def _compute_shares(self,row:pd.Series,ticker:str, inv:float) -> list[int,float]:
        # function returns a list with shares and price executed

        # When executing, we always cross the spread.
        # make sure we don't have fractional number of contracts
        if inv>0.0:
            price = row['ask_price'][ticker]
        else:
            price = row['bid_price'][ticker] 
            
        return int(inv/price),price
    
    def _create_pnl_graph(self,acc_df:pd.DataFrame) -> None:
        acc_df['cum_PnL'].plot()
        plt.title(f'Cumulative PnL (ewma_halflife={self.ewma_halflife} seconds)')
        plt.xlabel('Time')
        plt.ylabel('PnL(USD)')
        plt.legend()
        plt.show()

    def _create_inv_graph(self,acc_df:pd.DataFrame) -> None:
        acc_df['pos_value'].plot()
        plt.title(f'Investments (ewma_halflife={self.ewma_halflife} seconds)')
        plt.xlabel('Time')
        plt.ylabel('Investment(USD)')
        plt.legend()
        plt.show()
    
    def _create_pnl_hist(self,acc_df:pd.DataFrame) -> None:
        acc_df['PnL'].hist()
        plt.title(f'Histogram PnL(second)')
        plt.xlabel('PnL(usd)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


    def run_simulation(self) -> None:
        # the returned list will be containing two dataframes:
        # 1. Money_position_value : [date, symbol, money, pos_value]
        # 2. Trade report: [date, symbol, volume, price, buy?]
        self.data['signal'] = compute_signal(self.data, self.ticker_y, self.ticker_x,
                                             self.ewma_halflife,self.ewma_min_periods,
                                             adjust_ewma=self.ewma_adjust)
        i=0
        for date,row in self.data.iterrows():
            
            # The trading logic is pretty simple.
            # 1.Compute Optimal_investment based on the signal

            optimal_investment = self._compute_opt_investment(row.signal.item())
            #TO-DO: Add conditions for bid/ask quotes and volumes
            #can_trade = self._can_trade(row)
            

            if optimal_investment!=np.nan and optimal_investment!=self.curr_inv and abs(optimal_investment)<self.max_inv:
                #if can_trade:
                # trade amount can be >0 or <0
                trade_amount = optimal_investment-self.curr_inv
                
                # In order to avoid the engine to be rebalancing for very
                # low quantities, check trade_amount is higher than a certain amount
                if abs(trade_amount) > self.min_inv_factor*self.inv_per_trade:
                    print('rebalancing!')

                    # Split the investment and compute amount of shares
                    inv_y = trade_amount/2.0
                    inv_x = -1.0*trade_amount/2.0
                    
                    y_to_trade, self.price_y = self._compute_shares(row,self.ticker_y,inv_y)
                    x_to_trade, self.price_x = self._compute_shares(row,self.ticker_x,inv_x)

                    self.trades.append([date,self.ticker_y,y_to_trade,self.price_y,inv_y>0.0])
                    self.trades.append([date,self.ticker_x,x_to_trade,self.price_x,inv_x>0.0])

                    # update our global positions with the shares(contracts) traded
                    self.pos_y += y_to_trade
                    self.pos_x += x_to_trade

                    # if we are buying, cash gets out of this account
                    # if we are selling, cash enters this account
                    # thus the - sign in the inv
                    self.money_y += -1.0*y_to_trade*self.price_y
                    self.money_x += -1.0*x_to_trade*self.price_x

                    # Finally update the current investment
                    self.curr_inv = optimal_investment
                
            # update balances to keep track of portfolio value
            # to do so, use the mid price
            # Get the prices to valuing our positions
            self.ref_price_y = row['mid_price'][self.ticker_y].item()
            self.ref_price_x = row['mid_price'][self.ticker_x].item()

            self.accounts.append([date,self.ticker_y,self.money_y,self.pos_y*self.ref_price_y])
            self.accounts.append([date,self.ticker_x,self.money_x,self.pos_x*self.ref_price_x])
            
            i+=1
            print(f'Progress...{round(i/self.data.shape[0],4)}')

        # Convert lists into numpy arrays
        #return self.trades, self.accounts
    
    def analyze_results(self) -> dict[tuple,float]:
        # from self.trades and self.accounts, compute some statistics. Convert those
        # lists into numpy arrays to make easier the creation of the dataframe
        accounts_arr = np.array(self.accounts)
        trades_arr = np.array(self.trades)
                              
        accounts_df = pd.DataFrame(accounts_arr[:,1:],columns=['symbol', 'money', 'pos_value'],index=accounts_arr[:,0])
        trades_df = pd.DataFrame(trades_arr[:,1:],columns=['symbol','volume','price','buy?'],index=trades_arr[:,0])

        # Generate PnL statistics from the accounts dataframe
        accounts_df['balance'] = accounts_df['money']+accounts_df['pos_value']
        # First row will always be Nan because there's no previous row, fill it with 0
        accounts_df['PnL'] = accounts_df.groupby('symbol')['balance'].diff()
        accounts_df['PnL'].fillna(0,inplace=True)

        
        accounts_df['cum_PnL'] = accounts_df.groupby('symbol')['PnL'].transform(pd.Series.cumsum)
        # Create a dataframe with only one observation per symbol
        agg_accounts_df = accounts_df.groupby(level=0).agg({'symbol':'sum','pos_value':'sum','balance':'sum','PnL':'sum','cum_PnL':'sum'})
        
        # Compute overall statistics from both dataframes
        overall_PnL = sum(accounts_df['PnL'])
        total_trades = len(trades_df)
        total_turnover = sum(abs(trades_df['volume'])*trades_df['price'])
        avg_trade_size = np.mean(abs(trades_df['volume'])*trades_df['price'])
        rot = overall_PnL/total_turnover # return over turnover
        

        # Create a PnL graph, an investment graphs and return the statistics
        # Also plot a histogram with the distribution of PnLs
        self._create_pnl_hist(agg_accounts_df)
        # PnL graph
        self._create_pnl_graph(agg_accounts_df)
        # Investments graph
        self._create_inv_graph(agg_accounts_df)
        
        statistics = [overall_PnL,total_trades,total_turnover,avg_trade_size,rot]

        return statistics,trades_df,accounts_df



        # compute distribution of hourly PnLs
        # since we don't have a lot of data, at most a few
        # days worth of test data, I think it's
        # more interesting to analyze the distribution
        # of PnLs aggregating them per hours
        
        #
        #hourly_samples = agg_accounts_df.loc[((agg_accounts_df.index.minute==0)&(agg_accounts_df.index.second==0)),:]
        # compute mean metrics such as the mean PnL and Std_dev
        
        

#csv_file = '/home/axelbm23/Code/webb_traders_assignement/ESNQ.csv'
csv_file = '/home/axel/Downloads/ESNQ.csv'

# we will split our data into 30% for training, 30% for cross validation
# and 40% for out-of-sample testing
analysis_settings = {'train_split':0.3,
                    'crossval_split':0.6}
y_ticker = 'ESc1'
x_ticker = 'NQc1'
train_df,cv_df,test_df = import_data(csv_file,analysis_settings)

##################### RESEARCH SECTION ##########################
# As it is widely known, pair trading strategies are based
# on a mathematical property called cointegration. Simply put, two 
# time series are said to be cointegrated if a linear combination
# of them is mean reverting. In order to test this, we are going
# to run an Ordinary-least-square in the log of prices
def compute_OLS(df:pd.DataFrame):
    y_prices = np.log(df[y_prices])

compute_OLS(train_df)

##################### PARAMETERS ESTIMATION #####################

# We'll use a bunch of different halflife, from more reactive to less
# reactive for our analysis. 
ewma_halflifes = [3600]#,1800,3600,7200]

quantiles : dict[int,float]= {}

for ewm_hl in ewma_halflifes:
    # Compute the signal for each ewma and get the quantiles.
    signal = compute_signal(train_df,y_ticker,x_ticker,ewm_hl,ewm_hl*0.5)
    # In order to reduce complexity in our analysis, we will assume 
    # symmetrical and unskewed distributions of the signal around it's mean.
    quantiles[ewm_hl] = compute_dev_width(signal,[0.75,0.9])

    # Generate a plot that illustrates the signal and entry,exit points chosen
    # Note pandas.plot interpolates the points from day to another, leading
    # to very poor graphs.
    def plot_signal(signal:pd.Series,qtles:list[float]):
        idx = pd.date_range(start=min(signal.index),end=max(signal.index),freq='s')
        temp_df = pd.DataFrame(index=idx)
        temp_df['signal'] = signal
        fig,ax = plt.subplots()
        ax.plot(range(temp_df.dropna().size), temp_df.dropna())
        plt.axhline(y=qtles[0],color='r',linestyle='--',label='first entry point')
        plt.axhline(y=qtles[1],color='g',linestyle='--',label='max entry point')
        ax.set_xticklabels(temp_df.dropna().index)
        fig.autofmt_xdate()
        plt.show()



# now run our analysis for all the settings considered
# in the cross validation set and store the main statistics
# in a dictionary
statistics : dict[str,list[float]] = {}

for ewma_id,qtile in quantiles.items():
    sett = create_settings(ewma_id,qtile[0],qtile[1])
    bcktest = BackTester(sett,cv_df.iloc[:int(cv_df.shape[0]*0.3),:])
    bcktest.run_simulation()
    statistics[(ewma_id)],trades,accounts = bcktest.analyze_results()

# from all the possibilities, choose the one delivering the best results
# and test it out of sample


#sett = create_settings
#bcktest = BackTester(sett,train_df)
#trades,accounts = bcktest.run_simulation()




# TO-DO: Run OLS, justify that ratio should be Y/X instead of
# Y/beta*x

# TO-DO: Run dicky fuller cointegration test to asses whether
# Y and X are actually cointegrated.

# TO-DO: For the testing phase, compute the signal with multiple decay
# factors and compute the min and max deviation using percentiles on
# the distribution of 

# TO-DO: For the cross validaton phase, run the sim with the parameters
# previously computed and analyze the overall performance of the strategy


# TO-DO: For the test phase, run the algorithm with the "best" setups
# previously computed and generate again the trade analysis.



# How do we define the spread? We basically have three options here
# 1. spread=log(y/x)-ewma(log(y/x)) => Params: Half life of the ewma


    

    




# 1. Data exploration and analysis. In this section the data
# will be explored and common questions
# and some exponatory analysis will be conducted on the signal.

# 2. Backtesting. Generate all the metrics needed for the post
# analysis such as trades, account_balances and overall PnL and
# investment.
# 3. Produce post trade analysis.
 



# 1. Check what's the average wavelength of the signal to revert,
# conduct some hypothesis testing on the data to know whether the
# variables are cointegrated.
#test_df['ratio'].ewm