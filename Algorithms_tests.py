# This package is devoted to check that Algorithms 
# are correctly implemented

import numpy as np
import pandas as pd
from Algorithms import GradDescLR


def Test_Grad_Desc(epsilon:float, true_estimates:np.array, plot_cost:bool) -> bool:
    # If the error on the estimates if lower than epsilon, we'll assume
    # that Test is passed (True), else it will be False

    #  This will generate around 200 rows of data
    # with the first column being a column of ones
    # and the other two random numbers in the interval [-1.0,1.0]
    x = np.c_[np.ones(shape=(200,1)),2*np.random.rand(200,2)-1.0]

    # Create a synthetic dependent varible, Y, which
    # will be a predefined combination of the features (Y = 2.0+15.0*x1+25*x2)
    # and the true estimates
    y = np.dot(x,true_estimates)

    # The idea is that Gradient Descent will get to those coeficients
    grad_desc = GradDescLR(x=x,y=y,iter=1000)
    # The first component of w should be the intercept term, (~17)
    w,costs = grad_desc.Compute_Coefficients()
    discrepancies = w/true_estimates - 1.0
    
    if any(discrepancies>epsilon):
        return False
    
    if plot_cost:
        pd.Series(costs).plot(xlabel="# iteration", ylabel="J(w)",title="Gradient Descent: Cost function vs Iterations")

    return True

tests = []
tests.append(Test_Grad_Desc(epsilon=0.01, true_estimates=np.array([10,15,25]), plot_cost=True))
print(f"Out of the {len(tests)} tests,{sum(tests)} have passed")






