"""
Analysis file which will be calling
utils file to perform the two tasks
required: Construct the OB from
single updates and create some alphas
from the order book
"""

import argparse
import csv
import glob
import os
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import sys, importlib
import time
from utils import orderbook,generate_header,price_tick
importlib.reload(sys.modules['utils'])


# TO_DO_2: Re-organize/refactorize code in 
#          generate_orderbooks function

def create_output_folder() -> None:
    # TO-DO: Create output folder according to command line option
    if not os.path.isdir('generated_ob'):
        os.mkdir('generated_ob')

def generate_orderbooks(log:bool) -> None:
    print("Starting construction of orderbooks...")
    start_time = time.time()
    files = glob.glob(args.path)
    if os.listdir('generated_ob')==[] or args.regenerate_ob:
        create_output_folder()
        for f_idx,file in enumerate(files):
            # Clear the orderbook for each file (deleting all orders from the orderbook)
            ob = orderbook(log_warnings=log)
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
                        line_to_write = ob.format_output()
                        writer.writerow(line_to_write)
                wf.close()
            rf.close()
            print(f"Progress...{(f_idx+1)/len(files)}")
    
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

def compute_ofi(df:pd.DataFrame, levels:int) -> int:
        
        """
        Function devoted to compute order flow imabalance
        
        level: Specifies which levels the order flow information
                will be computed on 
        """
        for level in levels:
            
            org_qties = [f'bp{level}',f'bq{level}',f'ap{level}',f'aq{level}']
            shft_qties = [f'shifted_{label}' for label in org_qties]

            df[shft_qties] = df[[f'bp{level}',f'bq{level}',f'ap{level}',f'aq{level}']].shift()
            
            # TO-DO: Homogenize this so that we don't have to loop two times
            delta_bid = df.apply(lambda x: x[f'shifted_bp{level}'] if pd.isna(x[f'shifted_bp{level}']) else
                                 x[f'bq{level}'] if x[f'bp{level}']>x[f'shifted_bp{level}']
                                else x[f'bq{level}']-x[f'shifted_bq{level}'] if x[f'bp{level}']==x[f'shifted_bp{level}']
                                else -1*x[f'shifted_bq{level}'],axis=1)
            
            delta_ask = df.apply(lambda x: x[f'shifted_ap{level}'] if pd.isna(x[f'shifted_ap{level}']) else 
                                 -1*x[f'shifted_aq{level}'] if x[f'ap{level}']>x[f'shifted_ap{level}']\
                                else x[f'aq{level}']-x[f'shifted_aq{level}'] if x[f'ap{level}']==x[f'shifted_ap{level}']
                                else x[f'aq{level}'],axis=1)
            
            df[f'OFI_lvl_{level}'] = delta_bid-delta_ask
            
            df.drop(shft_qties,axis=1,inplace=True)

        return df

def generate_alphas_and_targets(df:pd.DataFrame,
                                lb_periods:list[int],
                                fw_window:list[int],
                                levels:list[int]) -> pd.DataFrame:
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

    # First compute the variable to be predicted
    for fwd in fw_window:
        # Note the negative side in -fwd because
        # it's a forward looking quantity, movement
        # in ticks in this case
        shifted_mids = df['mid_price'].shift(-fwd)
        df[f'fut_price_change_{fwd}'] = (shifted_mids-df['mid_price'])/(0.5*price_tick)
    
    df = compute_ofi(df,levels)

    # Compute order flow imbalance information for the periods
    # specified in lb_periods
    for level in levels:
        for lb in lb_periods:
            # In order to compute OFI for each period, we have to first compute OFI
            # for two consecutive updates. Then the OFI of multiple periods will be
            # just the sum of the individual OFI
            df[f'OFI_{lb}_lvl_{level}'] = df[f'OFI_lvl_{level}'].rolling(lb).sum()
    
    print(f"Computation time for alphas/targets={round(time.time()-start_time,3)}s")

    return df


def subsample(df:pd.DataFrame, gr:float) -> pd.DataFrame:
    """
    Gr: Defines the amount of seconds over which the data
    will be aggregated.
    """
    org_size = len(df)
    resolution = gr*1_000_000
    df['timestamp'] = df['timestamp']//resolution
    df['timestamp'] = df['timestamp'].astype(int)
    df = df.groupby(['timestamp']).last().reset_index()

    equally_sampled_ts = [ts for ts in range(max(df['timestamp'])+1)]
    aux_ts = pd.DataFrame(equally_sampled_ts,columns=['timestamp'])

    df_sampled = pd.merge(aux_ts,df,how='left',left_on='timestamp',right_on='timestamp')
    # For the sampling periods in which there were no updates, propagate the previous
    # update
    df_sampled.ffill(inplace=True)

    print(df_sampled.head())

    print(f"Orderbook size has been changed by {round(len(df_sampled)/org_size,3)}")

    return df_sampled

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
    # TO-D0: Before performing any regression,
    # make sure that the ranges of predictory
    # variables are similar. If not, 
    # perform some regularization
    
    lasso = linear_model.Lasso(alpha=alpha)
    lasso.fit(X=df[predictors], y=df[target])
    coefs = lasso.coef_
    intercept = lasso.intercept_
    score = lasso.score(X=df[predictors], y=df[target])

    return coefs,intercept,score

def run_ols(df:pd.DataFrame,predictors:list[str],
            target:str) -> tuple[list[float],list[float],float]:
    
    lreg = linear_model.LinearRegression().fit(X=df[predictors], y=df[target])
    coefs = lreg.coef_
    intercept = lreg.intercept_
    score = lreg.score(X=df[predictors], y=df[target])

    return coefs,intercept,score


def prepare_for_regression() -> pd.DataFrame:
    ob_files = glob.glob(f'{"/".join(args.path.split("/")[:-2])}/generated_ob/out*.csv')
    formatted_files = []
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
        
        # Drop columns as price and side which are not relevant anymore
        ob.drop(['price','side'],axis=1,inplace=True)

        # TO-DO: Pass sampling period as a command line argument?? Not sure
        # I want to allow this
        ob_sampled = subsample(ob,1.0)

        ob_sampled = generate_alphas_and_targets(ob_sampled, lb_periods=args.lb_periods,
                                                 fw_window=args.fw_periods, levels=args.levels)
        formatted_files.append(ob_sampled)
    
    return pd.concat(formatted_files)

def format_regression_results(results:dict, predictors:list[str]) -> pd.DataFrame:
    """
    Function that performs data wrangling over the result of
    the regression analysis and produces a readable dataframe
    """
    results_df = pd.DataFrame.from_dict(results, orient='index')
    predictors_labels = [f'coef_{label}' for label in predictors]
    results_df[predictors_labels] =  results_df[0].apply(lambda x: pd.Series(x))
    # Remove the auxiliary column 0
    results_df.drop(0,axis=1,inplace=True)

    results_df.reset_index(inplace=True)
    results_df.rename(columns={'index':'settings', 1:'intercept',2:'r_square'},inplace=True)

    return results_df

def run_regression(df:pd.DataFrame) -> pd.DataFrame:
    """
    This function is devoted to generate a set of statistics to determine
    which of the regressive models delivers the better results.
    In order to do so, an output csv will be generated with all
    the desired statistics

    df: input dataframe containing all predictors and target quantities
    alpha: parameter controlling the stength of the regularization
    predictors: list of labels to try to predict target
    target: future quantity that needs to be predicted
    """

    pred = ['vol_imbalance','ba_spread']
    pred.extend([f'OFI_lvl_{lvl}' for lvl in args.levels])
    pred.extend([f'OFI_{lb}_lvl_{lvl}' for lb in args.lb_periods for lvl in args.levels])
    target = [f'fut_price_change_{period}' for period in args.fw_periods]

    # Prior to run any regression, deal with Nans
    org_size = len(df)
    df.dropna(inplace=True)
    print(f'Dataset has been reduced by '
          f'{round(100.0*(1.0-len(df)/org_size),3)}% '
          f'after NaN treatment')
    
    # TO-DO: Check range of variables

    results = {}
    for tgt in target:
        tgt_label = tgt.split('_')[-1]
        for alpha_i in args.alphas:
            params_label = "_".join([str(x) for x in args.lb_periods])
            if alpha_i==0.0:
                # The way Lasso algorithm is implemented in sklearn
                # does not ensure the convergence of gradient descent (its optimizer)
                # when alpha=0.0. It is for this reason that I am using
                # LinearRegression from sklearn to compute the regression
                results[(alpha_i,params_label,tgt_label)] = run_ols(df,pred,tgt)
            else:
                results[(alpha_i,params_label,tgt_label)] = run_lasso_regression(df,pred,tgt,alpha_i)

    # Create a dataframe with the results and save it via a csv file
    results_df = format_regression_results(results,pred)
    results_df.to_csv(f'{args.output_path}/Regression_statistics.csv')

    return results_df

parser = argparse.ArgumentParser(prog='Orderbook analyzer',
                                 description='Package devoted to create and analyze orderbooks')
parser.add_argument('-path', help = 'Absolute path containing the input files',
                    type = str, default = '/home/axelbm23/Code/AlgoTrading/orderbook/codetest/res_*.csv')
parser.add_argument('-regenerate_ob', help = 'If true script will regenerate the orderbooks from scratch, else '
                    'it will try to load them', type=bool, default=False)
parser.add_argument('-fw_periods', help = 'Array containing the forward looking periods '
                    'to compute mid price changes (in seconds)', nargs = '+', default = [1,5,10,60,300,600])
parser.add_argument('-lb_periods', help = 'Array containing the backward looking periods, '
                    'used to compute set of predictors (in seconds)', default = [5,10,60,300,600])
parser.add_argument('-train_size', help = 'Percentage of data devoted to train the model',
                    type = float, default = 0.7)
parser.add_argument('-levels', help = 'Array specifying which ob levels will be taken into account '\
                    'to compute the order flow imbalance over', nargs='+', default=[0])
parser.add_argument('-alphas', help= 'Regularization term used for the Lasso regression',
                    type=str, nargs='+', default=[0.0,0.2,0.4,0.6,0.8,1.0])
parser.add_argument('-output_path', help='Absolute path where the results of the regressions'\
                    'will be stored', type=str,default='/home/axelbm23/Code/AlgoTrading/orderbook/')
parser.add_argument('-ob_logger', help='If True, the script will print lots debugging information '\
                    'for the orderbook creation part', type=bool, default=False)
args = parser.parse_args()

generate_orderbooks(args.ob_logger)
sampled = prepare_for_regression()

# Split the data set into train and cross validation
# we use a random_state to make this split deterministic,
# otherwise it is randomized
sampled_train,sampled_test = train_test_split(sampled, train_size=args.train_size,random_state=10)
reg_results = run_regression(sampled_train)

# TO-DO: Include the regression stats in the zip file!
#       Comment on the poor performance of our predictions
#       so far, with the best approach getting a 5% r_square
#       in the future returns

# GUIDELINE FOR THE RESEARCH
#1) Play with the forward return, the lookback periods
#   and the regularization parameter



# Final checks => Make sure program runs removing the default
#                 command line parameters.  