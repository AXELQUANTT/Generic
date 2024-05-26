"""
Analysis file which will be calling
utils file to perform the two tasks
required: Construct the OB from
single updates and create some alphas
from the order book
"""

import glob
import csv
from utils import orderbook,generate_header
import sys, importlib
importlib.reload(sys.modules['utils'])
import time


# TO-DO_1: Use command line options to run the script
# TO_DO_2: Re-organize/refactorize code in 
#          generate_orderbooks function
# TO_DO_3: Deal with in/out files in a better way. Probably
#          better to create an output folder storing the
#          ob files

def generate_orderbooks(path:str) -> None:
    files = glob.glob(path)
    k = 0
    print("starting construction of orderbooks...")
    for file in files:
        # define all file specific variables
        ob = orderbook()
        # Create the corresponding output file
        full_path = file.split('/')
        out_file = f"{'/'.join(full_path[:-1])}/out_{full_path[-1].split('_')[-1]}"
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
        k += 1
        print(f"Progress...{k/len(files)}")

start_time = time.time()
path = "/home/axelbm23/Code/AlgoTrading/orderbook/codetest/res_*.csv"
generate_orderbooks(path)
end_time = time.time()
print(f"Total execution time(s)={end_time-start_time}")

# Part 2)
# 2.1 Come up with a set of statistics that according to research
#     have some sort of predicitive analysis
# 2.2 Calculate the predective features from the orderbooks of
#     task 1
# 2.3 Create a prediction target that you think it would be
#     useful for trading the product. The  most straightforward
#     approach would be the 1m, 2m, 10m mid return
# 2.4 Subsample data? 
# 2.5 Perform Lasso on the subset of what we think are predictors
#     of the mid return of the orderbook. For those features
#     for which we have a coefficient very close to 0, we
#     can then infer that are not very relevant, so we can effectively
#     remove them from our model