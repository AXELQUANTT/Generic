import pandas as pd
import numpy as np
import argparse
from typing import Any
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

def create_settings(ewma_hl,min_spread,max_spread) -> dict[dict[str,Any]]:

    signal_settings : dict = {'ewma_halflife':ewma_hl,#in seconds
                            'ewma_adjust':False,#do not correct for ewma weights.
                            'ewma_min_periods':ewma_hl*0.5}

    order_settings : dict = {'min_spread':min_spread,
                            'max_spread':max_spread}
    
    investment_settings : dict = {'inv_per_trade':250_000}

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
    df = df.pivot(index='date',columns='symbol',values=['bid_price','bid_size',
                                                        'ask_price','ask_size','mid_price'])

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

    # test that train_df, cv_df and test_df do not overlap
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

def compute_signal(df,y_ticker,x_ticker,beta,alpha,ewma_hl,min_periods,adjust_ewma=False) -> pd.Series:
    ratio = np.log(df['mid_price'][y_ticker]) - beta*np.log(df['mid_price'][x_ticker]) - alpha
    ewma = ratio.ewm(halflife=ewma_hl,adjust=adjust_ewma,min_periods=min_periods).mean()
    signal = ratio-ewma

    return signal

def plot_signal(data:pd.Series,title:str,y_label:str,constants:list[float]):
    data_to_plot = pd.DataFrame(data,columns=[y_label])
    data_to_plot.index = data_to_plot.index.astype(str)
    data_to_plot[y_label].plot(rot=45)
    
    for cnst in constants:
        plt.axhline(y=cnst,color='g',linestyle='--')
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('time')
    plt.grid()
    plt.show()


def compute_OLS(x:pd.Series,y:pd.Series,halflife:bool) -> list[float,pd.Series]:

    x = sm.add_constant(x)
    model = sm.OLS(y,x)
    results = model.fit()

    # Get the params we are interested, mainly beta and alpha
    # and compute the error.
    alpha = results.params.values[0]
    beta = results.params.values[1]
    errors = y - (beta*x.iloc[:,1]+alpha)

    hl=0.0
    if halflife:
        hl = -np.log(2)/beta

    return alpha,beta,errors,hl

def compute_signal_hl(timeseries:pd.Series):
    lagged = timeseries.shift(1)
    lagged.iloc[0] = 0

    ret = timeseries-lagged
    ret.iloc[0] = 0

    _,_,_,hl_errors = compute_OLS(y=ret,x=lagged,halflife=True)

    return hl_errors

def check_stationarity(errors) -> None:
    plot_signal(errors,f'OLS errors\n {y_ticker} - ({round(beta,3)}*{x_ticker} + '
                f'{round(alpha,3)})','errors',[np.mean(errors)])
    
    # As it can be seen from the graph above, residuals of the 
    # of the OLS do not seem stationary. Nevertheless, we will make
    # use of the Dickey-Fuller to test if they have a unit root (=mean reverting)

    # Compute Dickey-Fuller test on the residuals and retrieve its p-value
    # This is actually the formal way to check whether the residuals have a unit
    # root, or in other words, whether they are stationary or not.
    # Ho: Null Hypothesis: data is non-stationary
    # Ha: Alternative Hypothesis: data is stationary
    print('Computing Dickey-Fuller test for cointegration...')
    confidence_interval = 0.05
    adfuller_pval = ts.adfuller(errors)[1]
    # In this case the p-value of the test is much larger than the usual 1% or
    # 5% confidence interval normally used, so we failed to reject our null hypothesis
    # and therefore residuals are non-stationary
    print(f'adfuller(pvalue) = {adfuller_pval} > {confidence_interval}')
    print(f'{y_ticker} and {x_ticker} are not cointegrated')

def report_results(stats:pd.DataFrame,trades:pd.DataFrame,accounts:pd.DataFrame,name:str) -> None:
    stats_df = pd.DataFrame.from_dict(stats,orient='index',
                                        columns=['overall_PnL','total_trades',
                                                 'total_turnover','avg_trade_size','rot(bps)'])
    stats_df.reset_index(inplace=True)
    stats_df[['emwa_hl(s)','min_spread','max_spread']] = stats_df['index'].apply(lambda x: pd.Series(x))
    # Remove index column as we don't need it anymore
    # And write the statistics into a csv
    stats_df.drop('index',axis=1,inplace=True)
    stats_df.to_csv(f'Overall_stats_{name}.csv',index=False)

    if not trades.empty:
        trades.to_csv(f'All_trades_{name}.csv')
    if not accounts.empty:
        accounts.to_csv(f'Accounts_{name}.csv')

class BackTester():
    def __init__(self, settings:dict[dict[str,Any]], data:pd.DataFrame):
        # Set some state variables used for the simulation
        self.data = data
        self.ticker_y = y_ticker
        self.ticker_x = x_ticker
        
        # Account level objects. All these dictionaries
        # are used to keep track of current investments, positions, etc
        self.curr_pos : dict [str,int] = {self.ticker_y:0,
                                          self.ticker_x:0}
        
        self.money : dict [str,float]= {self.ticker_y:0.0,
                                          self.ticker_x:0.0}
        
        self.ref_prices : dict [str,float] = {self.ticker_y:np.nan,
                                              self.ticker_x:np.nan} # reference prices
        
        self.opt_inv : dict [str,float] = {self.ticker_y:0.0,
                                           self.ticker_x:0.0} # optimal_investment
        
        self.curr_inv : dict[str,float] = {self.ticker_y:0.0,
                                           self.ticker_x:0.0} # current investment
        self.tot_inv = 0.0

        # The two objects where the results of the backtest will be stored.
        self.trades = []
        self.accounts = []
        
        # Investment Settings
        self.inv_per_trade = settings['inv_sett']['inv_per_trade']

        # Signal settings
        self.ewma_halflife = settings['signal_sett']['ewma_halflife']
        self.ewma_adjust = settings['signal_sett']['ewma_adjust']
        self.ewma_min_periods = settings['signal_sett']['ewma_min_periods']
        
        # Order settings
        self.min_spread = settings['order_sett']['min_spread']
        self.max_spread = settings['order_sett']['max_spread']


    def _simple_rebalancing(self,signal:float) -> float:
        if pd.isna(signal):
            return signal
        else:
            # Close position
            if abs(signal) < self.min_spread:
                return 0.0
            # Keep position
            elif self.max_spread >= abs(signal) >= self.min_spread:
                return self.tot_inv
            # Set up position
            else:
                optimal_investment = -1.0*self.inv_per_trade if signal >0 else self.inv_per_trade
                return optimal_investment
    
    def _compute_shares(self,row:pd.Series,ticker:str, inv:float) -> list[int,float]:
        # function returns a list with shares and price executed

        # Assumption. To simplify the execution part of the algorithm
        # we will assume that we get executed at mid price
        if inv>0.0:
            price = row['mid_price'][ticker]
        else:
            price = row['mid_price'][ticker] 
        
        # make sure we don't have fractional number of contracts    
        return int(inv/price),price


    def run_simulation(self) -> None:
        # This function simulates a backtest run on the previously
        # initialized self.data dataframe. Albeit not returning anything,
        # it does update internally self.trades and self.accounts,
        # the two objects we will need to analyze the results of the backtest
        self.data['signal'] = compute_signal(self.data, self.ticker_y,self.ticker_x,
                                             beta,0.0,self.ewma_halflife,
                                             self.ewma_min_periods,
                                             adjust_ewma=self.ewma_adjust)
        
        for date,row in self.data.iterrows():
            
            signal = row.signal.item()

            # Get the ref prices for valuing our positions
            self.ref_prices[self.ticker_y] = row['mid_price'][self.ticker_y].item()
            self.ref_prices[self.ticker_x] = row['mid_price'][self.ticker_x].item()
            
            # Update curr_inv (per symbol) and total investment
            self.curr_inv[self.ticker_y] = self.curr_pos[self.ticker_y]*self.ref_prices[self.ticker_y]
            self.curr_inv[self.ticker_x] = self.curr_pos[self.ticker_x]*self.ref_prices[self.ticker_x]
            self.tot_inv = self.curr_inv[self.ticker_y] - self.curr_inv[self.ticker_x]

            # Compute Optimal Investment
            optimal_investment = self._simple_rebalancing(signal)
            
            if not pd.isna(optimal_investment) and optimal_investment!=self.tot_inv:
                for symb in [self.ticker_y,self.ticker_x]:

                    symb_opt_inv = optimal_investment/2 if symb==self.ticker_y else -1.0*beta*optimal_investment/2.0
                    to_trade = symb_opt_inv - self.curr_inv[symb]
                    shs_traded,price_traded = self._compute_shares(row,symb,to_trade)
                    
                    if shs_traded!=0:

                        #print('rebalancing!')
                        self.trades.append([date,symb,shs_traded,price_traded,shs_traded*price_traded,
                                            shs_traded>0.0,signal])
                        # update our global positions with the shares(contracts) traded
                        self.curr_pos[symb] += shs_traded
                        # if we are buying, cash gets out of this account
                        # if we are selling, cash enters this account
                        # thus the - sign in the inv
                        self.money[symb] += -1.0*shs_traded*price_traded

            # Update the accounts dataframe with all the relevant data fo both symbols
            self.accounts.append([date,self.ticker_y,self.money[self.ticker_y],
                                  self.curr_pos[self.ticker_y]*self.ref_prices[self.ticker_y],
                                signal,self.min_spread,self.max_spread])
            self.accounts.append([date,self.ticker_x,self.money[self.ticker_x],
                                  self.curr_pos[self.ticker_x]*self.ref_prices[self.ticker_x],
                                signal,self.min_spread,self.max_spread])
    
    def analyze_results(self) -> dict[tuple,float]:
        # from self.trades and self.accounts, compute some statistics. Convert those
        # lists into numpy arrays to make easier the creation of the dataframe
        accounts_arr = np.array(self.accounts)
        accounts_df = pd.DataFrame(accounts_arr[:,1:],columns=['symbol','money','pos_value',
                                                               'signal','min_spread','max_spread'],
                                                               index=accounts_arr[:,0])
    

        # Generate PnL statistics from the accounts dataframe.
        # Cumulative PnL is just the same of money and pos_value
        accounts_df['cum_PnL'] = accounts_df['money']+accounts_df['pos_value']
        # First row will always be Nan because there's no previous row, fill it with 0
        # Period PnL will be just the difference of Cum_PnL
        accounts_df['PnL'] = accounts_df.groupby('symbol')['cum_PnL'].diff()
        accounts_df['PnL'].fillna(0,inplace=True)

        # Create a dataframe with only one observation per symbol
        agg_accounts_df = accounts_df.groupby(level=0).agg({'symbol':'sum','pos_value':'sum',
                                                            'PnL':'sum',
                                                            'cum_PnL':'sum','signal':'first',
                                                            'min_spread':'first',
                                                            'max_spread':'first'})
        
        # Compute overall statistics
        overall_PnL = sum(accounts_df['PnL'])

        trades_arr = np.array(self.trades)
        if trades_arr.size!=0:
            trades_df = pd.DataFrame(trades_arr[:,1:],columns=['symbol','volume','price',
                                                               'value','buy?','signal'],
                                                               index=trades_arr[:,0])
            avg_trade_size = np.mean(abs(trades_df['volume'])*trades_df['price'])
            total_trades = len(trades_df)
            total_turnover = sum(abs(trades_df['volume'])*trades_df['price'])
            rot = overall_PnL*10000.0/total_turnover # PnL over turnover in bps

            # Make sure that no trades are happening if spread is between min and max
            check = trades_df.loc[((abs(trades_df['signal']) >= self.min_spread)&
                                   (abs(trades_df['signal']) <= self.max_spread)),:]
            if not check.empty:
                print('CHECK SHOULD BE EMPTY AND IT IS NOT!!!')
        else:
            trades_df = null_df
            avg_trade_size = 0.0
            total_trades = 0.0
            total_turnover = 0.0
            rot = 0.0
            
        # Round ewma to print it in the graphs
        ewma_label = round(self.ewma_halflife,3)
        
        # Create a plot with the signal and the trades that have occurred during the sim
        entry_exit_points = [-1.0*self.max_spread,-1.0*self.min_spread,self.min_spread,self.max_spread]
        plot_signal(self.data['signal'],f'Signal (ewma={ewma_label}s) with entry/exit points',
                    'signal',entry_exit_points)

        # PnL graphs
        # First Graph shows overall cumulative PnL
        # Second graph shows PnL for ticker_y
        # Third graph shows PnL for ticker_x
        print('Printing PnLs of simulations ')
        plot_signal(data=agg_accounts_df['cum_PnL'],
                    title=f'Cumulative Overall PnL, ewma_halflife={ewma_label}s'
                     f', entry_spread={self.max_spread}, exit_spread={self.min_spread}',
                    y_label='cum_PnL',
                    constants=[])
        ticker_y_pnl = accounts_df.loc[accounts_df['symbol']==self.ticker_y,'cum_PnL']
        plot_signal(data=ticker_y_pnl,
                    title=f'Cumulative PnL, symbol={self.ticker_y}, ewma_halflife={ewma_label}s'
                     f', entry_spread={self.max_spread}, exit_spread={self.min_spread}',
                    y_label='cum_PnL',
                    constants=[])
        ticker_x_pnl = accounts_df.loc[accounts_df['symbol']==self.ticker_x,'cum_PnL']
        plot_signal(data=ticker_x_pnl,
                    title=f'Cumulative PnL, symbol={self.ticker_x}, ewma_halflife={ewma_label}s'
                     f', entry_spread={self.max_spread}, exit_spread={self.min_spread}',
                    y_label='cum_PnL',
                    constants=[])
        
        ticker_y_inv = accounts_df.loc[accounts_df['symbol']==self.ticker_y,'pos_value']
        ticker_x_inv = accounts_df.loc[accounts_df['symbol']==self.ticker_x,'pos_value']
        
        # Investment Graphs
        # First graph shows the Position imbalance. In a real world example,
        # executions on both sides of the pair may not happen at the same time
        # Therefore, one common risk measure is to see how big imbalances in the pair
        # are (imbalance=pos_y + pos_x).
        # Second graph shows the spread positions, inv_y - inv_x
        # Third and fourth graph show the Investment on each symbol independently.
        print('Printing investment of simulations ')
        plot_signal(data=agg_accounts_df['pos_value'],
                    title=f'Position Imbalance (inv_y + inv_x), ewma_halflife={ewma_label}s'
                     f', entry_spread={self.max_spread}, exit_spread={self.min_spread}',
                    y_label='pos_value',
                    constants=[])
        plot_signal(data=ticker_y_inv-ticker_x_inv,
                    title=f'Spread position (inv_y - inv_x), ewma_halflife={ewma_label}s'
                     f', entry_spread={self.max_spread}, exit_spread={self.min_spread}',
                    y_label='pos_value',
                    constants=[])
        plot_signal(data=ticker_y_inv,
                    title=f'Investment(USD) in symbol={self.ticker_y}, ewma_halflife={ewma_label}s'
                    f', entry_spread={self.max_spread}, exit_spread={self.min_spread}',
                    y_label='pos_value',
                    constants=[])
        plot_signal(data=ticker_x_inv,
                    title=f'Investment(USD) in symbol={self.ticker_x}, ewma_halflife={ewma_label}s'
                     f', entry_spread={self.max_spread}, exit_spread={self.min_spread}',
                    y_label='pos_value',
                    constants=[])
        
        statistics = [overall_PnL,total_trades,total_turnover,avg_trade_size,rot]

        return statistics,trades_df,accounts_df
        
# Create argparser to read the file from the command line
parser = argparse.ArgumentParser(
                    prog='Futures_PairTrading',
                    description='Program generates a pair trading strategy with post trade analysis',
                    epilog='Enjoy the analysis')

parser.add_argument('input_data',
                    help='Path containing the CSV with the futures data',
                    type=str,
                    nargs='?')

args = parser.parse_args()

# we will split our data into 50% for training, 30% for cross validation
# and 20% for out-of-sample testing
null_df = pd.DataFrame([])
analysis_settings = {'train_split':0.5,
                    'crossval_split':0.8}
y_ticker = 'ESc1'
x_ticker = 'NQc1'

train_df,cv_df,test_df = import_data(args.input_data,analysis_settings)

##################### COINTEGRATION ANALYSIS ####################
# The first part of the analysis will be devoted to study whether
# the two variables of the analysis are indeed co-integrated.
# Two variables are said to be co-integrated if a linear
# combination of them is mean reverting. An ordinary least
# square regression will be run over the log of prices, the 
# errors will be plotted and then a Dickey Fuller test will be 
# run over the errors to check if they are stationary.

print(f'Computing OLS to to retrieve hedge_ratio (beta)...')
alpha,beta,errors,_ = compute_OLS(y = np.log(train_df['mid_price'][y_ticker]),
                                x = np.log(train_df['mid_price'][x_ticker]),
                                halflife=False)
print(f'Spread computed. log(y) = {round(beta,3)}*log(x) + {round(alpha,3)}')
check_stationarity(errors)

################# MODEL & PARAMETERS ESTIMATION ################# 
# In terms of our model, our spread will be defined as 
# = log(y) - beta*log(x). For each usd inverted in Y, we are going
# to invest -1.0*beta in x.
# Our trading signal will be computed as spread - ewma(spread).
# Note that for that ewma, we will need to specify its halflife.
# In order to compute that ewma, we will run an autoregression 
# model on our signal (errors) to estimate what is the time it takes
# to that signal to mean revert.

# Once we have a proxy for the ewma used, we will compute our
# entry/exit signals based on percentiles of the spread. 
# These entry-exit points will be computed from the train data 
# set (train_df), and will be used in unseen data (cross 
# validation set, cv_df) to simulate some trading results.
# Finally, the best parameter configuration will be tested 
# in unseen data, test_df. Remember the ewma computed above
# is a good proxy, but different configurations around that
# level will be tested

hl_signal = compute_signal_hl(errors)
print(f'Half_life of signal(hours) = {hl_signal/3600.0}')

testing_settings = [(0.8*hl_signal,0.6,0.85),
                    (0.8*hl_signal,0.75,0.9),
                    (hl_signal,0.6,0.85),
                    (hl_signal,0.75,0.9),
                    (1.25*hl_signal,0.6,0.85),
                    (1.25*hl_signal,0.75,0.9)]


# dict to store the value of the quantiles
quantiles : dict[tuple[float],float] = {}

for test_set in testing_settings:
    # Compute the signal for each ewma and get the quantiles.
    ewma_hl = test_set[0]
    min_qtile = test_set[1]
    max_qtile = test_set[2]

    signal = compute_signal(train_df,y_ticker,x_ticker,beta,0.0,ewma_hl,ewma_hl*0.5)
    # In order to determine our entry/exit levels, compute percentiles of the spread
    # We will use symmetrical entry points for shorting/going long the spread, 
    quantiles[test_set] = compute_dev_width(signal,[min_qtile,max_qtile])
    negative_entry_points = [-1.0*x for x in quantiles[test_set]]
    
    # The trading logic is kept fairly easy
    # If abs(spread) > max_spread, we will set up a position.
    # if max_spread > abs(spread) > min_spread, we will keep the 
    #                               position (if any)
    # if abs(spread) < min_spread, we will close the position.
    

    # Generate a plot that illustrates the signal and entry,exit points chosen
    # according to percentiles of data. First green line shows the take
    # profit (min_spread), second green line shows the initiate trade
    # signal (max_spread)
    print('Following plot shows the entry-exit points in the spread')
    plot_signal(data=signal.dropna(),title=f'signal = spread - ewma(spread) \n' 
                f'spread = log({y_ticker})-{round(beta,3)}*log({x_ticker}) \n'
                f'Green lines are entry {quantiles[test_set][1]}'
                f' and exit points {quantiles[test_set][0]}',
                y_label='trading_signal',
                constants=quantiles[test_set]+negative_entry_points)
    
statistics : dict[tuple[float],list[float]] = {}
i=0
n_sims = len(quantiles.keys())
print('Starting simulations...')
for ind_sett,qtile in quantiles.items():
    # ewma_id is a float, but since it is in seconds  
    # and we do not have sub-second resolution in our data, 
    # discretize it to int.
    ewma_id = int(ind_sett[0])
    min_spread = qtile[0]
    max_spread = qtile[1]

    sett = create_settings(ewma_id,min_spread,max_spread)
    bcktest = BackTester(sett,cv_df)
    bcktest.run_simulation()

    statistics[(ewma_id,min_spread,max_spread)],_,_ = bcktest.analyze_results()
    i+=1
    print(f'Progress running sims on cv_data...{i}/{n_sims}')
    
# Save into a csv file all the results from the backtest
report_results(statistics,null_df,null_df,'cross_validation')
print('Statistics for the cross validation data are saved!')

# From all the possibilities, choose the one delivering
# the best results and test it out of sample.
# Basic criteria => choose the strategy with highest 
# return over turnover
best_configuration = list(statistics.keys())[0]
max_rot = list(statistics.values())[0][-1]

for config,values in statistics.items():
    new_rot = values[-1]
    if new_rot > max_rot:
        best_configuration = config

##################### OUT OF SAMPLE TESTING #####################
# Test the best configuration from the cv_test out of sample
print(f'Running out of sample simulation')
best_ewma = best_configuration[0]
min_spread = best_configuration[1]
max_spread = best_configuration[2]
# Save the stats into a dictionary with the settings
oos_stats : dict[tuple[float],list[float]] = {}

out_of_sample_sett = create_settings(best_ewma,min_spread,max_spread)
out_of_sample_bcktest = BackTester(sett,test_df)
out_of_sample_bcktest.run_simulation()
oos_stats[(best_ewma,min_spread,max_spread)],oos_trades,oos_accounts = out_of_sample_bcktest.analyze_results()

report_results(oos_stats,oos_trades,oos_accounts,'out_of_sample')
print('Statistics for the out of sample test are saved!')
print('Analysis has ended!')

# Just as an end note. The whole point of this script has been
# illustrating a pipeline of research and development. Here
# below are a bunch of improvements that could be done
# to the algorithm. Some of them come as limitations 
# from such a reduced sample size. 

# hedge ratio (=beta). Should be dynamically computed,
# with our approach it is assumed that it is constant
# throughout the entire period. 

# small testing data sample. All the parameters
# needed for the strategy (entry/exit points and ewma halflfe)
# are computed from data for comprising just a 
# few days. In a real world example, these parameters 
# should be much more robust and obtained over a bigger sample
# period.

# better optimization. The proposed settings have not been
# finely curated for the seek of a better overall results (mainly
# entry,exit percentiles) but just picked as a 'reasonable'
# choice.

# better execution analysis. For simplicity,
# executions at the mid price have been considered in this analysis.
# No transaction or slippage costs have therefore been accounted
# for.
