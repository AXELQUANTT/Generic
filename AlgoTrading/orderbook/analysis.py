"""
Analysis file which will be calling
utils file to perform the two tasks
required: Construct the OB from
single updates and create some alphas
from the order book. This file only contains
the command line arguments that modify
the analysis
"""

import argparse
import os
from sklearn.model_selection import train_test_split
import sys, importlib
from utils import *
importlib.reload(sys.modules['utils'])

#sys.argv= ['']
parser = argparse.ArgumentParser(prog='Orderbook analyzer',
                                 description='Package devoted to create and analyze orderbooks')
parser.add_argument('--path', help = 'Absolute path containing the input files',
                    type = str, default =f'{os.path.abspath(os.getcwd())}/codetest/res_*.csv')
parser.add_argument('--regenerate_ob',
                    help = 'If true script will regenerate the orderbooks from scratch, else it will try to load them '\
                    'if they were previously created in generated_ob',
                    type=bool, default=True)
parser.add_argument('--fwd_periods',
                    help = 'Array containing the forward looking periods to compute mid price changes (in seconds)',
                    default=[5,10,60,300,600,1800,3600,7200])
parser.add_argument('--lb_periods', 
                    help = 'Array containing the backward looking periods used to compute set of predictors (in seconds)',
                    default=[5,10,60,300,600,1800,3600,7200])
parser.add_argument('--train_size', help = 'Percentage of data devoted to train the model',
                    type = float, default = 0.6)
parser.add_argument('--levels', help = 'Array specifying which ob levels will be taken into account '\
                    'to compute the order flow imbalance over', default=[0])
parser.add_argument('--alphas', help= 'Regularization term used for the Lasso regression',
                    type=str, default=[0.0,1.0,10,100,10_000])
parser.add_argument('--ob_logger', help='If True, the script will print lots debugging information '\
                    'for the orderbook creation part', type=bool, default=False)
parser.add_argument('--sampling_window', help = 'Sampling period (in seconds) over which the original orderbook '\
                    'will be aggregated. Smallest possible value of this is 0.5',
                    type=float, default=1.0)
args = parser.parse_args()

generate_orderbooks(args.ob_logger, args.path, args.regenerate_ob)
sampled = prepare_for_regression(args.path,args.sampling_window,
                                 args.lb_periods,args.fwd_periods,
                                 args.levels)

# Split the data set into train,cross_validation and test
sampled_train,cv_and_test = train_test_split(sampled,train_size=args.train_size,shuffle=False)
sampled_cv,sampled_test = train_test_split(cv_and_test,train_size=0.5,shuffle=False)

# Run regressions and get the best explained mid-price change 
# and the best parameters and model to predict it
reg_results,results,models = run_regression(sampled_train, args.levels,
                                            args.lb_periods, args.fwd_periods,
                                            args.alphas)
fwd_ret,r_sq = get_most_explanable_return(reg_results)
fwd_price_label = f"fut_price_change_{fwd_ret}"
best_model = get_best_model(sampled_cv, models,
                            fwd_price_label, fwd_ret,
                            args.levels, args.lb_periods)

# Finally, once we have selected the best model with our training
# and cross_validation set, the last piece is to generate a
# trading signal and plot it
data_with_signal, out_sample_r_square = construct_signal(sampled_test, best_model,
                                                         fwd_price_label, args.levels,
                                                         args.lb_periods)

print_signal(data_with_signal, fwd_price_label)

print('Analysis concluded')