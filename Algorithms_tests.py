# This package is devoted to check that Algorithms 
# are correctly implemented

import numpy as np
import pandas as pd
from Algorithms import *
from sklearn import linear_model

def Test(true_params:np.array, test_params:np.array, epsilon:float) -> bool:
    discrepancies = test_params/true_params - 1.0
    
    if any(discrepancies>epsilon):
        return False
    
    return True 

tests = []

# Linear Regression Gradient Descent without intercept term
# The algo should compute parameters equal to 15,25 for X1 and X2
x = 2*np.random.rand(200,2)-1.0
parameters = np.array([15,25])
y =  np.dot(x,parameters)
comp_params,cost_log = GradDescLinReg(x,y,iter=1000,intercept=False).Compute_Coefficients()
tests.append(Test(parameters,comp_params,0.01))


# Linear Regression Gradient Descent with intercept term
x = np.c_[np.ones(shape=(200,1)),x]
parameters = np.array([10,15,25])
y = np.dot(x,parameters)
comp_params,cost_log = GradDescLinReg(x,y,iter=1000,intercept=True).Compute_Coefficients()
tests.append(Test(parameters,comp_params,0.01))


# Logistic Regression without regularization
# SK-learn applies regularization by default to its parameters,
# so we can compare the parameters by our algo with and without regularization
# and we should see they're smaller when regularizing
x_train = np.array([[1,0.5,1.5], [1,1,1], [1,1.5,0.5], [1,3,0.5], [1,2,2], [1,1,2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
comp_params,cost_log = GradDescLogReg(x=x_train,y=y_train,iterations=10000,intercept=True,alpha=0.1).Compute_Parameters()

# This data is taken from the coursera course, so we know the true values we should get here
theo_params = np.array([-14.222409982019837, 5.28, 5.08])
tests.append(Test(theo_params,comp_params,0.01))

# A good sanity check is to ensure that the parameters obtained with 
# regularization are smaller than the ones obtained without regularization
reg_comp_params,cost_log = GradDescLogReg(x=x_train,y=y_train,iterations=10000,intercept=True,lambda_=10.0).Compute_Parameters()
tests.append(all(abs(reg_comp_params)<abs(comp_params)))

print(f"Passed test? {all(tests)}!")
print(f"Out of the {len(tests)} tests,{sum(tests)} have passed")