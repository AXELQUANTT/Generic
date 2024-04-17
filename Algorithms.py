# The whole idea of this package is to create multiple classes, where
# each class will be a different Machine Learning algorithm

# Exercise 1. Implement the class gradient descent
import numpy as np
from typing import Tuple
import warnings


class GradDescLR:

    def __init__(self, x:np.array, y:np.array, iter:int) -> None:
        # Here x and y are the features and targets of the Training Set!
        self.x = x
        self.y = y
        self.obs,self.feat = x.shape
        self.iter = iter

    # Note that input below can be any training data or unseen (test) data
    def Predict(self,input:np.array, parameters:np.array) -> np.array:
        n_params = len(parameters)

        if n_params != self.feat:
            raise Exception (f"parameters={parameters} have the wrong size")
                
        if n_params != input.shape[1]:
            raise Exception (f"Wrong parameter <=> input shape, {n_params}(n_params)!={input.shape[1]}(input_features)")

        # This is an array, where each column represents the i-th prediction
        # for input in x
        return np.dot(input,parameters)
    
    def _Cost(self, parameters:np.array) -> float:
        
        # Prediction seems to be okay
        predictions = self.Predict(self.x, parameters)
        
        if predictions.shape!=self.y.shape:
            raise Exception (f"targets.shape({self.y.shape})!=predictions.shape({predictions.shape})")
        
        errors_sq = (predictions-self.y)**2

        return (0.5/self.obs)*np.sum(errors_sq)

    def _Derivatives(self, parameters) -> Tuple[np.array,float]:
        
        predictions = self.Predict(self.x, parameters)
        errors = predictions-self.y

        # Check that first component of self.x.T is a column with only ones
        param_der = (1/self.obs)*np.dot(self.x.T,errors)

        # Check derivatives has the correct dimensions
        if param_der.shape!=(self.feat,):
            raise Exception ("Something went wrong, derivatives has incorrect size!")
        
        return param_der

    def Compute_Coefficients(self) -> Tuple[np.array, list]:
        
        alphas = [0.0001,0.0003,0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0][::-1]

        for alpha in alphas:
            # Start gradient descent with generic values (1s for instance)
            w = np.ones(self.feat)
            k = alpha/self.obs
            prev_cost = self._Cost(w) # Cost is correct
            finalized=True
            log_costs = list()
            #print(f"computing gradient descent for alpha={alpha}")
            for i in range(self.iter+1):
                
                # Compute gradient and update parameters
                w_dev = self._Derivatives(w)
                w -= k*w_dev

                # Compute cost for each w and b
                cost = self._Cost(w)

                if cost>prev_cost:
                    warnings.warn("Cost is increasing, alpha is too big")
                    finalized=False
                    break
                prev_cost = cost
                log_costs.append(cost)
            
            # Since we are starting with the highest alpha, the first alpha
            # that makes gradient descent converge is the most optimal one,
            # so we don't need to iterate over the rest
            if finalized:
                print(f"Gradient Descent has finished for alpha={alpha}")
                return (w,log_costs)
        # if we get here is because none of the alphas made gradient descent to converge
        # in which case raise a program error
        raise ValueError(f"None of the alphas={self.alphas} makes Gradient Descent converge, check")