"""
Analysis file which will be calling
utils file to perform the two tasks
required: Construct the OB from
single updates and create some alphas
from the order book
"""










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