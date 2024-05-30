"""
Utilities package containing all the
assistant functions needed to generate
the orderbook. This package also
contains functions and methods for the
second part of the exercise, which
is devoted to create some alphas for
the aggregated orderbooks
"""

import csv
import glob
from heapq import heapify,heappop
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn import linear_model
import time
from typing import Tuple, List
import warnings


# Define some global variables
n_levels = 5
price_tick = 5

def generate_n_levels_labels() -> list[str]:
    """
    Aux function to create orderbook levels labels
    """
    labels = []
    for side in ['a','b']:
        for i in range(n_levels):
            labels.extend([f'{side}p{i}',f'{side}q{i}'])
    return labels

def generate_ob_derived_metrics() -> list[str]:
    return ['mid_price']

def generate_header() -> list[str]:
    """
    Aux function to create df headers
    """

    header = ['timestamp','price','side']
    header.extend(generate_n_levels_labels())
    header.extend(generate_ob_derived_metrics())

    return header

def create_output_folder() -> None:
    """
    Auxiliary function to create output folder if not already created
    """
    if not os.path.isdir('generated_ob'):
        os.mkdir('generated_ob')

def generate_orderbooks(log:bool, path:str, regenerate_ob:bool) -> None:
    """
    Function that reads the input files and generates the orderbooks
    """

    print("Starting construction of orderbooks...")
    start_time = time.time()
    files = glob.glob(path)
    create_output_folder()  
    if os.listdir('generated_ob')==[] or regenerate_ob:
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
    print(f'Orderbooks were created in {round(end_time-start_time,3)} s')

def compute_ofi(df:pd.DataFrame, levels:int) -> pd.DataFrame:
        """
        Function devoted to compute order flow imabalance
        """

        for level in levels:
            
            org_qties = [f'bp{level}',f'bq{level}',f'ap{level}',f'aq{level}']
            shft_qties = [f'shifted_{label}' for label in org_qties]

            df[shft_qties] = df[[f'bp{level}',f'bq{level}',f'ap{level}',f'aq{level}']].shift()
            
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
    """

    start_time = time.time()

    # Compute the variable to be predicted
    for fwd in fw_window:
        # Note the negative side in -fwd because
        # it's a forward looking quantity, movement
        # in ticks in this case
        shifted_mids = df['mid_price'].shift(-int(fwd))
        df[f'fut_price_change_{fwd}'] = (shifted_mids-df['mid_price'])/(0.5*price_tick)
    
    df = compute_ofi(df,levels)

    # Compute order flow imbalance information for the periods
    # specified in lb_periods and level in levels
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
    Function devoted to create an equally sized time window
    dataframe from the original one
    """

    org_size = len(df)
    resolution = gr*1_000_000
    df['timestamp'] = df['timestamp']//resolution
    df['timestamp'] = df['timestamp'].astype(int)
    df = df.groupby(['timestamp']).last().reset_index()

    equally_sampled_ts = [ts for ts in range(max(df['timestamp'])+1)]
    aux_ts = pd.DataFrame(equally_sampled_ts,columns=['timestamp'])

    df_sampled = pd.merge(aux_ts,df,how='left',left_on='timestamp',right_on='timestamp')
    # For the sampling periods in which there were no updates, 
    # propagate the previous update
    df_sampled.ffill(inplace=True)

    print(f"Orderbook size is {round(len(df_sampled)/org_size,3)} of what it was")

    return df_sampled

def run_lasso_regression(df:pd.DataFrame,
                         predictors:list[str],
                         target:str,
                         alpha:float=1.0) -> tuple[list[float],list[float],float,linear_model.Lasso]:
    """
    Functiond devoted to compute Lasso linear
    regression. 
    """

    lasso = linear_model.Lasso(alpha=alpha,fit_intercept=False)
    lasso.fit(X=df[predictors], y=df[target])
    coefs = lasso.coef_
    intercept = lasso.intercept_
    score = lasso.score(X=df[predictors], y=df[target])

    return coefs,intercept,score,lasso

def run_ols(df:pd.DataFrame,
            predictors:list[str],
            target:str) -> tuple[list[float],list[float],float, linear_model.LinearRegression]:
        
    """
    Functiond devoted to compute OLS regression.
    """
    
    lreg = linear_model.LinearRegression(fit_intercept=False).fit(X=df[predictors], y=df[target])
    coefs = lreg.coef_
    intercept = lreg.intercept_
    score = lreg.score(X=df[predictors], y=df[target])

    return coefs,intercept,score,lreg


def prepare_for_regression(path:str,
                           samp_window:list[int],
                           lb_periods:list[int],
                           fwd_periods:list[int],
                           lvls:list[int]) -> pd.DataFrame:
    """
    Function that prepares the dataframes for the regression
    """

    ob_files = glob.glob(f'{"/".join(path.split("/")[:-2])}/generated_ob/out*.csv')
    formatted_files = []
    
    if samp_window<0.5:
            raise ValueError(f"sample_window option has to be > 0.5s, it is {samp_window}")
    
    for file in ob_files:
        ob = pd.read_csv(file)

        # Drop columns as price and side which are not relevant anymore
        ob.drop(['price','side'],axis=1,inplace=True)

        ob_sampled = subsample(ob,samp_window)

        ob_sampled = generate_alphas_and_targets(ob_sampled, lb_periods=lb_periods,
                                                 fw_window=fwd_periods, levels=lvls)
        formatted_files.append(ob_sampled)
    
    # Create a single dataframe with unique index, which can be dropped
    df = pd.concat(formatted_files).reset_index(drop=True)
    return df

def format_regression_results(results:dict, predictors:list[str]) -> pd.DataFrame:
    """
    Function that performs data wrangling over the result of
    the regression analysis and produces a readable dataframe
    """
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.reset_index(inplace=True)
    results_df[['alpha','lb_periods','fwd_period']] = results_df['index'].apply(lambda x: pd.Series(x))

    predictors_labels = [f'coef_{label}' for label in predictors]
    results_df[predictors_labels] =  results_df[0].apply(lambda x: pd.Series(x))
    # Remove auxiliary columns
    results_df.drop([0,'index'],axis=1,inplace=True)
    results_df.rename(columns={1:'intercept',2:'r_square'},inplace=True)

    return results_df

def get_predictors(levels:list[int],
                   lb_periods:list[int]) -> list[str]:
    """
    Simple functions that generates the list of predictors labels
    """
    pred = []
    pred.extend([f'OFI_lvl_{lvl}' for lvl in levels])
    pred.extend([f'OFI_{lb}_lvl_{lvl}' for lb in lb_periods for lvl in levels])
    return pred


def run_regression(df:pd.DataFrame,
                   levels:list[int],
                   lb_periods:list[int],
                   fwd_periods:list[int],
                   alphas:list[float]) -> tuple[pd.DataFrame, dict, dict]:
    """
    This function is devoted to generate a set of statistics to determine
    which of the regressive models delivers the better results.
    In order to do so, an output csv will be generated with all
    the desired statistics
    """

    pred = get_predictors(levels,lb_periods)
    target = [f'fut_price_change_{period}' for period in fwd_periods]

    results = {}
    models = {}
    for tgt in target:
        tgt_label = tgt.split('_')[-1]
        for alpha_i in alphas:
            params_label = "_".join([str(x) for x in lb_periods])
            # Prior to run any regression, deal with Nans
            non_nans = df.dropna(subset=pred+[tgt])
            
            if alpha_i==0.0:
                # The way Lasso algorithm is implemented in sklearn
                # does not ensure the convergence of gradient descent (its optimizer)
                # when alpha=0.0. It is for this reason that I am using
                # LinearRegression from sklearn to compute the regression
                reg = run_ols(non_nans,pred,tgt)
                results[(alpha_i,params_label,tgt_label)] = reg[:3]
                models[(alpha_i,params_label,tgt_label)] = reg[-1]
            else:
                reg = run_lasso_regression(non_nans,pred,tgt,alpha_i)
                results[(alpha_i,params_label,tgt_label)] = reg[:3]
                models[(alpha_i,params_label,tgt_label)] = reg[-1]

    # Create a dataframe with the results and save it via a csv file
    results_df = format_regression_results(results,pred)
    results_df.to_csv(f'Regression_statistics.csv',index=False)

    return results_df,results,models

def get_most_explanable_return(results:pd.DataFrame) -> tuple[int,float]:

    """
    Function that takes as input the results of the regressions
    and gets the predicted feature, mid-price change, for which
    our regression have accomplished the most predective power
    """ 

    # From all the possible regressions, choose the fwd_period
    # thas has the biggest r_square
    mean_r_squares = results.groupby(['fwd_period']).agg({'r_square':'mean'})

    fwd_ret = mean_r_squares.idxmax().iloc[0]
    r_sq = mean_r_squares.max().iloc[0]
    
    print(f'The return that can be better predicted is in the '
      f'peridod {fwd_ret}, with mean r_square of {round(r_sq,3)}')
    
    return fwd_ret,r_sq

def get_best_model(df:pd.DataFrame,
                   models:dict,
                   fwd_price_label:str,
                   fwd_ret:str,
                   levels:list[int],
                   lb_periods:list[int]) -> tuple:
    """
    Function whose goal is to retrieve the best model
    out of the ones computed before
    """

    # Now that we know which mid price is the one that we can
    # predict better, we will compute the squared mean error
    # of all the models across the cross validation set
    # and we will choose the value of alpha that produces
    # the smallest mean_squared_error
    cv_df = df.copy()
    # Make sure the dataframe does not contain nans in the predictors/targets
    pred_labels = get_predictors(levels,lb_periods)
    cv_df.dropna(subset=pred_labels+[fwd_price_label],inplace=True)
    min_square_error = float("inf")
    best_model = ()
    for key,mdl in models.items():
        if key[-1]==fwd_ret:
            # Now predict and compute the mean_squared_error
            predicts = mdl.predict(cv_df[pred_labels])
            targets = np.array(cv_df[fwd_price_label])
            mean_square_error = sum((targets-predicts)**2)/len(targets)
            if mean_square_error<min_square_error:
                min_square_error = mean_square_error
                best_model = (key,mdl)
    
    print(f'The model with the lowest mean square error in '
          f'the cv set is {best_model}')
    
    return best_model

def construct_signal(df:pd.DataFrame,
                     best_model:tuple,
                     fwd_price_label:str,
                     lvls:list[int],
                     lb_periods:list[int]) -> tuple[pd.DataFrame, float]:
    """
    Function devoted to compute the trading signal into the test
    dataframe and report the r_square of the fit (out of sample)
    """
    # Create the signal on the test set and asses how good our fit
    # is out of the sample
    mdl = best_model[-1]
    pred_labels = get_predictors(lvls,lb_periods)
    test_no_nans = df.dropna(subset=pred_labels+[fwd_price_label])
    signal = pd.DataFrame(mdl.predict(X=test_no_nans[pred_labels]),index=test_no_nans.index,columns=['signal'])
    test_no_nans = test_no_nans.merge(signal,left_index=True,right_index=True)

    out_of_sample_r_square = best_model[1].score(X=test_no_nans[pred_labels], y=test_no_nans[fwd_price_label])

    print(f'The out of sample r_square value obtained with the best '
          f'model is {round(out_of_sample_r_square,3)}')

    return test_no_nans, out_of_sample_r_square

def print_signal(df:pd.DataFrame,fwd_ret:str) -> None:
    """
    Function devoted to print the signal generated from our regression
    analysis
    """

    for metric in ['signal',fwd_ret]:
        plt.plot(df[metric], label=metric)
    plt.ylabel(f'{fwd_ret}')
    plt.xlabel('time')
    plt.legend(loc='upper right')
    plt.title(f'predicted(blue) vs actual(orange)')
    plt.show()

# Why I am using a tuple instead of a list?
# We do not need to modify the incoming order book
# updates, so I've choosen it as a immutable data type
ob_update = Tuple[int,#timestamp
                  str,#side
                  str,#action
                  int,#id
                  float,#price. From the data seems like lot size is 5 monetary
                  # units, but in general it could be any floating point number
                  int, #volume, number of contracts bid/offered
                  ]
class orderbook:

    def __init__(self, log_warnings:bool=False) -> None:
        self.bid = {}
        self.ask = {}
        self.update : ob_update = ()
        # this dictionary keeps track of the current active
        # orders
        self.active_orders = {}
        self.log = log_warnings

    def _format_input(self, in_line:List) -> ob_update:
        ts,side,action,id,price,volume = in_line
        # By default all fields are read as strings
        # Change those that are not strings
        ts = int(ts)
        id = int(id)
        price = float(price)
        volume = int(volume)

        return ts,side,action,id,price,volume
    
    def _selector(self, side:str) -> dict:
        if side=='b':
            return self.bid
        elif side=='a':
            return self.ask
        else:
            # we should never get here, just a sanity check
            raise ValueError("side is not b or a, check!")
    
    def _add_orderid(self) -> None:
        ts,side,id,price,vol = self.update[0],self.update[1],self.update[3],self.update[4],self.update[5]
        # check if price is in the ob
        ob_side = self._selector(side)
        if price not in ob_side:
            ob_side[price] = [vol,set([id])]
        else:
            if self.log and id in ob_side[price][1]:
                warnings.warn(f"order_id={id} with timestamp={ts} was added to side={side} "
                                f"but it was already in the orderbook, seems weird!")
            # All this code should never be reached because an order that is already
            # in the orderbook should not be added for a second time. This was just
            # developed as control/sanity check
            ob_side[price][0] += vol
            ob_side[price][1].add(id)
        
        if self.log:
            print(f"Order_id={id} was added to side={side} with "
                  f"price={price} and vol={vol}")
        
        # Finally add the order to the dictionary of active orders
        self.active_orders[id] = [price,vol]         

    def _remove_orderid(self) -> None:
        side = self.update[1]
        ob_side = self._selector(side)
        id = self.update[3]
        price = self.update[4]
        vol = self.update[5]
        if price not in ob_side:
            raise ValueError(f"order_id={id} is deleted from side={side} but that price={price} "
                             f"was never in the orderbook")
        else:
            if id not in ob_side[price][1]:
                raise ValueError(f"price={price} is present in the ob but that "
                                 f"order_id={id} is not, CHECK!")
            else:
                ob_side[price][0] -= vol
                ob_side[price][1].remove(id)
                
                # Check that volumes are never negative
                if ob_side[price][0]<0:
                    raise ValueError(f"side={side}, price={price} has negative volume, vol={ob_side[price][0]}")
                
                if ob_side[price][0]==0:
                    # we should not have any order_id here
                    if ob_side[price][1]:
                        raise ValueError(f"there is no volume in this side={side}, price={price} "
                                         f"but there are some order_ids={ob_side[price][1]}")
                    else:
                        del ob_side[price]
                        if self.log:
                            warnings.warn(f"removing price={price} from orderbook")
                
        # Finally remove the order from the dictionary of active orders
        del self.active_orders[id]

    def _modify_orderid(self) -> None:
        id = self.update[3]
        if id not in self.active_orders:
            raise ValueError(f"order_id={id} wants to be modified but "
                             f"it's not in the list of active orders, CHECK!!")
        new_price,new_vol = self.update[4],self.update[5]
        old_price,old_vol = self.active_orders[id]
        side = self.update[1]
        ob_side = self._selector(side)
        if new_price==old_price and new_vol!=old_vol:
            # in this case we only need to modify the volume
            if self.log:
                warnings.warn(f"new_vol={new_vol}, old_vol={old_vol}, "
                            f"level_vol += {new_vol-old_vol}")
            ob_side[old_price][0] += new_vol-old_vol
            # Check that volumes are never negative
            if ob_side[old_price][0]<0:
                raise ValueError(f"side={side}, price={old_price} has negative " 
                                 f"volume, vol={ob_side[old_price][0]}")
            # only change the volume as price is the same
            self.active_orders[id][1] = new_vol

        elif new_vol==old_vol and new_price!=old_price:
            # volume is the same, but price has changed
            # In that case we need to remove the volume from
            # the old_price
            ob_side[old_price][0] -= new_vol
            ob_side[old_price][1].remove(id)

            # And add volume to the new price
            if new_price in ob_side[new_price]:
                ob_side[new_price][0] += new_vol
            else:
                ob_side[new_price] = [new_vol,set(id)]
            
            # only change the price as the volume is the same
            self.active_orders[id][0] = new_price

        else:
            raise ValueError("Price and volume have changed at the same time, check!")
    
    def process_update(self, update:list) -> None:
        self.update = self._format_input(update)
        action = self.update[2]

        match action:
            case 'a':
                # add_price_vol
                self._add_orderid()
            case 'd':
                # go to the specific level where the order sits and removes it
                self._remove_orderid()
            case 'm':
                # in case of a modification, according to the instructions
                # either price or vol have changed, but not both.
                self._modify_orderid()
    
    def _generate_statistics(self) -> None:

        # bid/ask prices can be None though, perfome checks:
        if self.ob_view['ap0'] and self.ob_view['bp0']:
            # mid_price => 0.5*(best_ask+best_bid)
            self.ob_view['mid_price'] = 0.5*(self.ob_view['ap0']+self.ob_view['bp0'])
        else:
            self.ob_view['mid_price'] = None
    
    def generate_ob_view(self) -> dict:

        self.ob_view = {'timestamp':self.update[0],
                        'price':self.update[4],
                        'side':self.update[1]}

        for side in ['a','b']:
            ob_side = self._selector(side)
            # Since we have to retrieve prices in sorted order,
            # one quick way to do it is via a heap
            # Note that heaps are min heaps in Python, so to get
            # a max heap(for the bid side) we need to insert negative prices,
            # which are of course converted again afterwards before being
            # distributed.
            price_vols = [[price,val[0]] if side=='a' else [-1*price,val[0]] for price,val in ob_side.items()]
            heapify(price_vols)
            
            for i in range(n_levels):
                if price_vols:
                    price,vol = heappop(price_vols)
                    self.ob_view[f"{side}p{i}"] = price if price>0 else -1*price
                    self.ob_view[f"{side}q{i}"] = vol
                else:
                    self.ob_view[f"{side}p{i}"] = None
                    self.ob_view[f"{side}q{i}"] = 0

        # Add orderbook derived statistics to the view
        self._generate_statistics()

        if self.log:
            print(f"order book view => {self.ob_view}")
        
        return self.ob_view
    
    def format_output(self) -> list:
        values = [self.ob_view[label] for label in generate_header()]
        return values