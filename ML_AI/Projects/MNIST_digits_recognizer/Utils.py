#import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.losses import SparseCategoricalCrossentropy
from keras.activations import relu, linear
from keras import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from scipy.ndimage import rotate
from collections import Counter

def compute_J(yhat, y) -> float:
    if len(yhat)!=len(y):
        raise Warning(f"len(yhat)={len(yhat)}  and len(y)={len(y)} do not have the same size!")
     
    # For the guys we incorrectly labeled, retrieve their distributions
    # to see if there's a particular number we missclassified more than
    # others (my theory is that numbers like 3-9 and 1-7 could be easily
    # misclassified as they're more similar)
    n_errors = sum(yhat!=y)
    abs_error = (1/len(yhat))*n_errors
    error_analysis = dict((key,value/n_errors) for key,value in Counter(y[y!=yhat]).items())

    acc = sum(yhat==y)/len(y)

    return abs_error, error_analysis, acc

# This function will return three outputs, the model itself and the costs in the train and cross validation datasets
def build_model_and_train(model_params, x_train, y_train, x_crossval, y_crossval, categories, n_epoch, batch_size):
    # This function is devoted to build and train a NN
    # and compute the Cost of its predictions on the train and
    # cross validation datasets

    if y_train.nunique()!=categories:
        raise Exception("the train dataset does not have examples of all categories")
    
    # Make sure the input passed to keras is correct
    x_t = np.array(x_train)

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

    model, j_train, j_cv, history, error_train, error_cv, acc_train, acc_cv = \
    train(model, x_train, y_train, x_crossval, y_crossval, n_epoch, batch_size)

    # Compute F1 score, which is actually the statistic that will
    # be used to rank our proposal
    
    return model, j_train, j_cv, history, error_train, error_cv, acc_train, acc_cv

def predict(model, x_in):
    # Now given an input, X, make predictions with our trained model
    # Softmax will return, for each row in x_t, a 1-D array of len=10
    # containing the probabilities of each number
    return np.argmax(tf.nn.softmax(model.predict(x_in, verbose=0)), axis=1)

def train(model, x_train, y_train, x_crossval, y_crossval, n_epoch, batch_size):

    x_t = np.array(x_train)
    y_t = np.array(y_train)
    x_cv = np.array(x_crossval)
    y_cv = np.array(y_crossval)

    history = model.fit(x_t, y_t, epochs=n_epoch, verbose=0, batch_size=batch_size)

    # Make predictions
    yhat_train = predict(model, x_t)
    yhat_cv = predict(model, x_cv)
    
    j_train, error_train, acc_train = compute_J(yhat_train,y_t)
    j_cv, error_cv, acc_cv = compute_J(yhat_cv,y_cv)

    return model, j_train, j_cv, history, error_train, error_cv, acc_train, acc_cv

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

def print_multiple_images(imag_arr,labels) -> None:
      # imag_arr is an array of array with n length, where each element is a 784 lenght array
      r = c = 0
      size = int(np.sqrt(len(imag_arr)))
      plt.figure()
      f,ax = plt.subplots(size,size)
      for idx,image in enumerate(imag_arr):
        ax[r][c].set_title(f"rot,noise={labels[idx][0],labels[idx][1]}")
        ax[r][c].get_xaxis().set_visible(False)
        ax[r][c].get_yaxis().set_visible(False)
        # Do not forget that the first value of the image is the label
        ax[r][c].imshow(np.array(image[1:]).reshape(28,28),cmap='gray', vmin=0, vmax=255)
        if c == 2:
          r += 1
          c = 0
        else:
           c += 1

def perform_data_agumentation(image,label,param,print_img=True) -> list:

    # param is an array of arrays, where the first component is the degrees to be
    # rotated in the image, while the second one regulates how much of an pixel distortion
    # we create

    # One quick way to create new images from the existing ones is to perform some rotations
    # on the existing images but making sure that the numbers are still readable by a human
    # (so cap the rotation to some sensible degrees)

    # Not only we will do that, but we will add some white noise over the pixels of the image
    # to make it a non linear combination of the input image

    result = []
    # Make random numbers predictable for now
    np.random.seed(0)
    for config in param:
      # Use order=1 to ensure value of pixels never exceeds 255, reshape=False to preserve
      # the original 28,28 size
      rotated = rotate(np.array(image).reshape(28,28),config[0],order=1,reshape=False).reshape(784)
      new_image = [label]+[int(pix*(1+config[1]*(np.random.rand(1)-1.0)/100.0)) if pix!=0 else pix for pix in rotated]
      result.append(new_image)

    # Print the created pictures for manually inspection
    # The idea here is that by altering the image pixels we
    # should make the algorithm to perform better on images
    # that are harder to classify
    if print_img:
        print_multiple_images(result,param)

    return result