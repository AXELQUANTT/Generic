"""
Package with some generic utils used across multiple
ML projects
"""

import tensorflow as tf
import tf_keras
import tf_keras.activations
from tf_keras.activations import relu,linear
import tensorflow as tf
from typing import Any
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
                    cost_tf_vars:Any,
                    optimizer:tf_keras.optimizers.Optimizer,
                    n_iters:int=100):
    
    for _ in range(n_iters):
        with tf.GradientTape() as tape:
            costJ = cost_tf_vars(tf_variables,x,y)
        
        derivates = tape.gradient(costJ, [tf_variables])

        optimizer.apply_gradients(zip(derivates,[tf_variables]))
    
    return tf_variables


def create_dense_layer(neurons:int,
                 activation:tf_keras.activations=relu,
                 l2_reg=None) -> tf_keras.layers.Dense:
    
    return tf.keras.layers.Dense(units=neurons, activation=activation, 
                  kernel_regularizer=l2_reg)

def create_nn(input_size: int,
              model_params: dict,
              output_size: int) -> tf_keras.Model:

    model = tf.keras.Sequential()
    if input_size!=0:
        model.add(tf.keras.Input(shape=(input_size,)))
    for neurons in model_params['neurons']:
        model.add(create_dense_layer(neurons=neurons,
                                     activation=relu))
    model.add(create_dense_layer(neurons=output_size, activation=linear))

    model.compile(loss=model_params['loss_function'],
                    optimizer=model_params['optimizer'])

    return model