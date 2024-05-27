"""
Analysis file which will be calling
utils file to perform the two tasks
required: Construct the OB from
single updates and create some alphas
from the order book
"""

import csv
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys, importlib
import time
from utils import orderbook,generate_header,price_tick
importlib.reload(sys.modules['utils'])


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

def compute_order_flow_imbalance(curr_update,prev_update) -> float:
    # Makes sense that if we are trying to predict future
    # price moves, we use order_flow_imbalance metrics
    # over time intervals on the same scale than the forward window
    None
    


def generate_alphas_and_targets(df:pd.DataFrame, fw_window:int) -> pd.DataFrame:
    """
    Function devoted to generate the target predictors
    and the alphas that will try to predict them
    
    forward_int: Mid price movement will be computed
                 over this interval (in ms)
    """

    #df.index = pd.DatetimeIndex(df['timestamp'])
    #df.rolling().apply()

    #TO-D0: Time this logic, I bet it's quite slow,
    #       code it in Pandas way

    alphas_targets = pd.DataFrame(index=df.index)
    for idx,row in df.iterrows():
        curr_mid = row['mid_price']
        curr_ts = row['timestamp']
        
        fwd_ts = df.loc[((df['timestamp'] >= curr_ts) & (df['timestamp'] <= curr_ts+fw_window)),'timestamp'].iloc[-1]
        fwd_mid = df.loc[df['timestamp']==fwd_ts,'mid_price']

        #TO-DO: Document why we are choosing the mid price change/tick_size and not
        #       the mid price return for instance       
        alphas_targets[idx,'mid_price_change'] = (fwd_mid-curr_mid)/price_tick


    return df


def subsample(df:pd.DataFrame, gr:str) -> pd.DataFrame:
    """
    Gr: Defines the minimum granularity over which the
    data will be aggregated. It does not ensure that the
    output dataframe will have that sampling period.
    Sampling period could be larger than that, but not smaller
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

    df.head()
    #Issue. We have a period in which there are no updatest that lasts for quite a while
    #       We want our price data to be equally sampled
    #       When computing the prices, instead of adding rows, just get the closest
    #       value. Is there an issue if we have a lot of values for which the price
    #       has not changed?
    #aux_ts = pd.DataFrame([ts for ts in range(max(df['timestamp'])+1)],columns=['timestamp'])

    #df_sampled = pd.merge(aux_ts,df,how='left',left_on='timestamp',right_on='timestamp')
    # For the sampling periods in which there were no updates, propagate the previous
    # update
    #df_sampled.ffill(inplace=True)

    print(f"Orderbook size has been reduced by {round(1-len(df)/org_size,3)}")

    return df

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
        ob_sampled = subsample(ob,'miliseconds')

        # TO-DO: Pass window size as a command line argument
        generate_alphas_and_targets(ob_sampled,10)


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
#     have some sort of predicitive analysis
# 2.2 Calculate the predective features from the orderbooks of
#     task 1.

# 2.3 Create a prediction target that you think it would be
#     useful for trading the product. The  most straightforward
#     approach would be the 1m, 2m, 10m mid return
#   2.4 Subsample data? Original updates are in microseconds since
#       the opening of the session. For sure we want to
#       aggregate all updates that happen in the same microsecond,
#       but shall we subsample more?       
# 2.5 Perform Lasso on the subset of what we think are predictors
#     of the mid return of the orderbook. For those features
#     for which we have a coefficient very close to 0, we
#     can then infer that are not very relevant, so we can effectively
#     remove them from our model