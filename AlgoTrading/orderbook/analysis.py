"""
Analysis file which will be calling
utils file to perform the two tasks
required: Construct the OB from
single updates and create some alphas
from the order book
"""

import csv
from sklearn import linear_model
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys, importlib
import time
from utils import orderbook,generate_header,price_tick
importlib.reload(sys.modules['utils'])
import warnings


# TO-DO_1: Use command line options to run the script
# TO_DO_2: Re-organize/refactorize code in 
#          generate_orderbooks function
# TO_DO_3: Deal with in/out files in a better way. Probably
#          better to create an output folder storing the
#          ob files

def create_output_folder() -> None:
    # TO-DO: Create output folder according to command line option
    if not os.path.isdir('generated_ob'):
        os.mkdir('generated_ob')

def generate_orderbooks(path:str) -> None:
    print("starting construction of orderbooks...")
    start_time = time.time()
    files = glob.glob(path)
    if os.listdir('generated_ob')==[]:
        create_output_folder()
        for f_idx,file in enumerate(files):
            # Clear the orderbook for each file (deleting all orders from the orderbook)
            ob = orderbook()
            # Create the corresponding output file
            full_path = file.split('/')
            out_file = f"{'/'.join(full_path[:-2])}/generated_ob/out_{full_path[-1].split('_')[-1]}"
            with open(file,"r") as rf:
                reader = csv.reader(rf,delimiter=',')
                # skip header
                next(reader,None)
                with open(out_file,"w") as wf:
                    writer = csv.writer(wf,delimiter=',')
                    # write the header to the output csv file
                    writer.writerow(generate_header())
                    for _,line in enumerate(reader):
                        ob.process_update(line)
                        ob.generate_ob_view()
                        line_to_write = ob._format_output()
                        writer.writerow(line_to_write)
                wf.close()
            rf.close()
            
        print(f"Progress...{f_idx+1/len(files)}")
    
    end_time = time.time()
    print(f'Orderbooks were created in {round(end_time-start_time,2)} s')

def compute_ofi_deltas(prev_update:pd.Series,
                       curr_update:pd.Series,
                       level:int=0) -> int:
    """
    Function devoted to compute the delta volumes for bid/ask prices
    needed for the order flow imbalance
    """
    # For convenience the first element will be the delta_bid
    # and the second element will be the delta_ask
    if prev_update.equals(curr_update):
        return 0
    else:
        delta_vol = {}
        for side in ['b','a']:
            
            delta_vol[side] = 0
            price_label = f"{side}p{level}"
            vol_label = f"{side}q{level}"

            curr_price = curr_update[price_label]
            curr_vol =  curr_update[vol_label]
            
            prev_price = prev_update[price_label]
            prev_vol = prev_update[vol_label]

            if curr_price > prev_price:
                delta_vol[side] = curr_vol if side=='b' else -1*prev_vol
            
            elif curr_price == prev_price:
                delta_vol[side] = curr_vol-prev_vol
            
            else:
                delta_vol[side] = -1*prev_vol if side=='b' else curr_vol
        
        return delta_vol['b']-delta_vol['a']

    

def compute_ofi(df:pd.DataFrame, curr_ts:int, lookback:list[int]) -> dict[int,int]:
    # Makes sense that if we are trying to predict future
    # price moves, we use order_flow_imbalance metrics
    # over time intervals on the same scale than the forward window
    
    """
    Function devoted to compute order flow imbalance (OFI). Order flow imbalance
    gives a measure not only about the intention of the participants in the
    market (i.e whether there is more buying than selling interest) but also
    about its magnitude.

    OFI_t = Delta_Vol_Bid_t - Delta_Vol_Ask_t
    
    Delta_Vol of bid and ask are computed via compute_ofi_deltas
    """
    
    # We'll create an array with OFI_t computed over 1ms intervals
    max_lookback = max(lookback)
    prev_ts = curr_ts-max_lookback
    
    # Since we have different lookback periods, we'll retrieve NaNs for
    # all the timestamps in which ANY of the lookback periods will be computed
    # The reason behind this is to ensure that our population of predictors
    # is equal in terms of size of valid values
    prev_update = df.loc[df['timestamp']<=prev_ts,['aq0','ap0','bq0','bp0']]
    # Generate multiple OFIs according to the different lookback periods
    of_lb = {}
    
    if prev_update.empty:
        warnings.warn(f'lookback periods can not be computed '
                      f'for timestamp={curr_ts}')
        ofi_arr = [np.nan]*max_lookback
    else:
        prev_update = prev_update.iloc[-1]
        ofi_arr = []
        # for i in range(1,max_lookback+1):
        #     next_ts = prev_ts+1
        #     if any(df['timestamp']==next_ts):
        #         next_update = df.loc[df['timestamp']==next_ts,['aq0','ap0','bq0','bp0']].iloc[0]
        #     else:
        #         # In case there was no update on the next ms
        #         # keep the old update 
        #         next_update = prev_update

        #     # OBSERVATION: THE ISSUE IS ON THE FOR LOOP, not on the computation of the OFI itself
        #     ofi = compute_ofi_deltas(prev_update, next_update, 0)
        #     ofi_arr.append(ofi)
            
        #     # # update prev_ts and update
        #     prev_update = next_update
        #     prev_ts += 1
    
    for lb in lookback:
        of_lb[f"OFI_period({lb})"] = sum(ofi_arr[:lb+1])

    return of_lb

def generate_alphas_and_targets(df:pd.DataFrame,
                                lb_periods:list[int],
                                fw_window:int) -> pd.DataFrame:
    """
    Function devoted to generate the target predictors
    and the alphas that will try to predict them
    
    forward_int: Mid price movement will be computed
                 over this interval (in ms)

    lb_periods: List containing the lookback periods
    over which order flow imbalance information will
    be computed
    """

    #TO-D0: Time this logic, I bet it's quite slow,
    #       code it in Pandas way. It is, before 
    #       the OFI computation it takes around 

    # Alphas_targets will be a dict of dicts, containing
    # all the predictors and targets to predict. Later
    # on, alphas_targets will be transformed into a df
    # to merge it with the original one

    start_time = time.time()
    alphas_targets = dict((ts,{}) for ts in df['timestamp'])
    for idx,row in df.iterrows():
        curr_mid = row['mid_price']
        curr_ts = row['timestamp']
        
        fwd_ts = df.loc[((df['timestamp'] >= curr_ts) & 
                         (df['timestamp'] <= curr_ts+fw_window)),
                        'timestamp'].iloc[-1]
        fwd_mid = df.loc[df['timestamp']==fwd_ts,'mid_price'].iloc[0]

        #TO-DO: Document why we are choosing the mid price change/tick_size and not
        #       the mid price return for instance
        alphas_targets[curr_ts]['mid_price_change'] = (fwd_mid-curr_mid)/(0.5*price_tick)

        alphas_targets[curr_ts].update(compute_ofi(df,curr_ts,lb_periods))
        print(f"Progress..{round(idx/len(df),5)}")
    
    print(f"Computation time for alphas/targets={round(time.time()-start_time,3)}s")
    
    alphas_targets_df = pd.DataFrame.from_dict(alphas_targets,orient='index').reset_index()
    alphas_targets_df.rename(columns={'index':'timestamp'},inplace=True)

    # What is the mean and median difference between updates though?
    mean_diff = np.nanmean(df['timestamp'].diff())
    median_diff = np.nanmedian(df['timestamp'].diff())
    print(f"mean_diff(ms)={mean_diff}, median_diff(ms)={median_diff}")

    # Do some quality checks to ensure df and alphas_targets_df have the same
    # timestamps
    if set(df['timestamp'])!=set(alphas_targets_df['timestamp']):
        raise ValueError('Data and target_predictors datasets do not ' 
                         'have the same timestamps')
    df = pd.merge(df, alphas_targets_df, left_on='timestamp', right_on='timestamp', how='left')

    return df


def subsample(df:pd.DataFrame, gr:str) -> pd.DataFrame:
    """
    Gr: Defines the minimum granularity over which the
    data will be aggregated. It does not ensure that the
    output dataframe will have that sampling period.
    Sampling period could be larger than that, but not smaller.
    """
    org_size = len(df)

    match gr:
        case 'raw':
            return df
        case 'microseconds':
            None
        case 'miliseconds':
            df['timestamp'] = df['timestamp']/1_000
        case 'deciseconds':
            df['timestamp'] = df['timestamp']*100/1_000
        case 'seconds':
            df['timestamp'] = df['timestamp']/1_000_000
    
    df['timestamp'] = df['timestamp'].astype(int)
    df = df.groupby(['timestamp']).last().reset_index()

    #df.head()
    #Issue. We have a period in which there are no updatest that 
    #       lasts for quite a while We want our price data to be
    #       equally sampled When computing the prices, instead of
    #       adding rows, just get the closest value. Is there an
    #       issue if we have a lot of values for which the price
    #       has not changed?
    #equally_sampled_ts = [ts for ts in range(min(df['timestamp'])+250)]
    #aux_ts = pd.DataFrame(equally_sampled_ts,columns=['timestamp'])

    #df_sampled = pd.merge(aux_ts,df,how='left',left_on='timestamp',right_on='timestamp')
    # For the sampling periods in which there were no updates, propagate the previous
    # update
    #df_sampled.ffill(inplace=True)

    #print(df_sampled.head())

    #print(f"Orderbook size has been changed by {round(len(df_sampled)/org_size,3)}")

    return df

def run_lasso_regression(df:pd.DataFrame,
                         predictors:list[str],
                         target:str,
                         alpha:float=1.0) -> tuple[list[float],list[float],float]:
    """
    Functiond devoted to compute Lasso linear
    regression. 
    
    df: Input dataframe containing the predictors
    , X, and the target to predict, y.

    alpha: Parameter that modulates the 
    strength of the regularization term 
    of the regression. When alpha=0,
    lasso regression is the usual OLS regression

    """
    
    lasso = linear_model.Lasso(alpha=alpha)
    lasso.fit(X=df[predictors], Y=df[target])
    coefs = lasso.coef_
    intercept = lasso.intercept_
    score = lasso.score(X=df[predictors], Y=df[target])

    return coefs,intercept,score


def analyse_orderbook() -> pd.DataFrame:
    # TO-DO: Read output orderbooks from command line option
    ob_files = glob.glob(f'{"/".join(path.split("/")[:-2])}/generated_ob/out*.csv')
    for file in ob_files:
        ob = pd.read_csv(file)
        #TO-DO: Develop criteria to select which sampling period is
        #       sensible. One quick way is to check the variance
        #       of quantities over the given sample period
        #       and discard those for which we have no
        #       variance on the predictive target.
        #       Another approach is to use multiple prediction targets and
        #       get the one with the better results for the
        #       Lasso regression.

        # TO-DO: Pass sampling period as a command line argument
        ob_sampled = subsample(ob,'seconds')

        #TO-DO: Pass fw_window and lb_periods via command line arguments
        #       so that user can modify them
        ob_sampled = generate_alphas_and_targets(ob_sampled, lb_periods=[5,10,50],fw_window=10)


        run_lasso_regression(ob_sampled)

        print(ob_sampled) 



path = '/home/axelbm23/Code/AlgoTrading/orderbook/codetest/res_*.csv'
#TO-DO: Add boolean command line argument to create orderbooks, even if they
#       already exist in the output folder. If True, already created ob files
#       will be read.

generate_orderbooks(path)
analyse_orderbook()


# For now the paramerters of the analysis are:
# granularity: over which the data is sampled, effect on end results should
# be pretty low
# foward_window: Can have more of an impactful effect on end result 

# Part 2)
# 2.1 Come up with a set of statistics that according to research
#     have some sort of predicitive analysis => Done
# 2.2 Calculate the predective features from the orderbooks of
#     task 1.

# 2.3 Create a prediction target that you think it would be
#     useful for trading the product. The  most straightforward
#     approach would be the 1m, 2m, 10m mid return => Done

#   2.4 Subsample data? Original updates are in microseconds since
#       the opening of the session. For sure we want to
#       aggregate all updates that happen in the same microsecond,
#       but shall we subsample more? => Done
       
# 2.5 Perform Lasso on the subset of what we think are predictors
#     of the mid return of the orderbook. For those features
#     for which we have a coefficient very close to 0, we
#     can then infer that are not very relevant, so we can effectively
#     remove them from our model