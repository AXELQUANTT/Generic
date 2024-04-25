# Repository for all the functions/methods used in our jupiter notebook
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.losses import SparseCategoricalCrossentropy
from keras.activations import relu, linear
from keras import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from keras.optimizers import Adam

def compute_J(yhat,y) -> float:
    if len(yhat)!=len(y):
        raise Warning(f"len(yhat)={len(yhat)}  and len(y)={len(y)} do not have the same size!")
    
    return (1/len(yhat))*sum(yhat!=y)

# This function will return three outputs, the model itself and the costs in the train and cross validation datasets
def build_model_and_train(model_params, x_train, y_train, x_crossval, y_crossval, categories):
    # This function is devoted to build and train a NN
    # and compute the Cost of its predictions on the train and
    # cross validation datasets

    if y_train.nunique()!=categories:
        raise Exception("the train dataset does not have examples of all categories")
    
    # Make sure the input passed to keras is correct
    x_t = np.array(x_train)
    y_t = np.array(y_train)
    x_cv = np.array(x_crossval)
    y_cv = np.array(y_crossval)

    # Build the architecture
    tf.random.set_seed(1234)
    model = Sequential()
    model.add(keras.Input(shape=(x_t.shape[1],)))
    for neurons in model_params['nn']:
        model.add(Dense(units=neurons, activation=relu, kernel_regularizer=tf.keras.regularizers.l2(model_params['lambda'])))

    # Finally add the output layer, which will be 10 neurons layer with linear activation function
    # We could specify a sigmoid function here, but there would be some rounding error with it
    # In order to avoid it, use SparseCategoricalCrossEntropy
    model.add(Dense(units=categories, activation=linear))

    # Configure the network => specify loss functiona and optimizer
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                    optimizer=Adam(learning_rate=0.001))
    
    # Finally train the model (perform backpropagation)
    # epoch specifies the iterations of the adam optimizer
    history = model.fit(x_t,y_t, epochs=25)

    print(model.layers)
    print(f"model summary = {model.summary()}")

    # Now given an input, X, make predictions with our trained model
    # Softmax will return, for each row in x_t, a 1-D array of len=10
    # containing the probabilities of each number
    yhat_train = np.argmax(tf.nn.softmax(model.predict(x_t)), axis=1)
    yhat_cv = np.argmax(tf.nn.softmax(model.predict(x_cv)), axis=1)
    
    j_train = compute_J(yhat_train,y_t)
    j_cv = compute_J(yhat_cv,y_cv)

    return model, j_train, j_cv, history

def plot_adam_cost(costs):
     fig,ax = plt.subplots()
     for i in range(len(costs)):
        ax.plot(np.linspace(1,len(costs[i][0]),len(costs[i][0])),
                 costs[i][0],label=costs[i][1])
     ax.set_xlabel("iterations")
     ax.set_ylabel("cost")
     ax.set_title("Adam optimizer cost vs Iterations")
     ax.grid()
     ax.legend()
     plt.show()