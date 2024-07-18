"""
Package with some utils that may be quite handy
for general ML_AI projects
"""

import keras
from keras import Sequential
from keras.activations import linear,relu
from keras.layers import Dense
#from keras.optimizers import Adam
import tensorflow as tf
import numpy as np

def compute_grad_desc(tf_variables:tf.Variable,
                        x:np.array,
                        y:np.array,
                        alpha:float=0.01,
                        n_iters:int=100) -> tf.Variable:
    """
    Function optimizes parameters using gradient
    descent via tensorflow (auto diff method).
    
    Note: tf_variables need to be initialized
    """
    
    for _ in range(n_iters):
        with tf.GradientTape() as tape:
            fwb = tf_variables*x
            costJ = (fwb-y)**2
        
        [derivates] = tape.gradient(costJ, [tf_variables])

        tf_variables.assign_add(-alpha * derivates)
    
    return tf_variables


# We can also think of the same concept but for a generic
# optimizer
def optimize_params(tf_variables:tf.Variable,
                    x:np.array,
                    y:np.array,
                    cost_tf_vars,
                    optimizer:keras.Optimizer,
                    n_iters:int=100):
    
    for _ in range(n_iters):
        with tf.GradientTape() as tape:
            costJ = cost_tf_vars(tf_variables,x,y)
        
        derivates = tape.gradient(costJ, [tf_variables])

        optimizer.apply_gradients(zip(derivates,[tf_variables]))
    
    return tf_variables
    
def create_nn(x:np.array,
              model_params:dict,
              output_size:int) -> keras.Model:

    # Build the architecture
    tf.random.set_seed(1234)
    model = Sequential()
    model.add(keras.Input(shape=(x.shape[1],)))
    for neurons in model_params['neurons']:
        model.add(Dense(units=neurons, activation=relu, kernel_regularizer=tf.keras.regularizers.l2(model_params['lambda'])))

    # Finally add the output layer
    model.add(Dense(units=output_size, activation=linear))

    # Configure the network => specify loss functiona and optimizer
    model.compile(loss=model_params['loss_function'], optimizer=model_params['optimizer'])

    return model

