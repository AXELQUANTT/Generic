"""
Unit test for utils package
"""

import keras
import numpy as np
import tensorflow as tf
import Utils

def Test(true_params:np.array, test_params:np.array, epsilon:float, binary:bool=False) -> bool:
    if not binary:
        discrepancies = test_params/true_params - 1.0
        # may be the case param is multidimensional
        for item in discrepancies.flatten():
            if item>epsilon:
                return False
        return True 
    else:
        discrepancies = true_params == test_params
        return sum(discrepancies)==len(discrepancies)

#####################################################################
############################### TESTS ############################### 
#####################################################################
tests = []

###########
# Grad desc
###########

w = tf.Variable(3.0)
x = [1.0]
y = [1.0]
iters = 1000

w_optimized = np.array(Utils.compute_grad_desc(tf_variables=w,x=x,y=y,n_iters=iters))
w_theo = np.array([1.0])

tests.append(Test(w_optimized,w_theo,0.001))

################
# Adam optimizer
################

w = tf.Variable(3.0)
x = [1.0]
y = [1.0]
iters = 1000
adam = keras.optimizers.Adam(learning_rate=0.1)
def costJ(w,x,y):
    return (w*x-y)**2

adam_optimized = np.array(Utils.optimize_params
                       (tf_variables=w,x=x,y=y,cost_tf_vars=costJ,
                        optimizer=adam,n_iters=iters))
adam_theo = np.array([1.0])

tests.append(Test(adam_optimized,adam_theo,0.001))

#####################################################################
############################### CHECK ############################### 
#####################################################################
print(f"Passed test? {all(tests)}!")
print(f"Out of the {len(tests)} tests,{sum(tests)} have passed")

