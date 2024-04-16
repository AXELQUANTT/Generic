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
    def Predict(self,input:np.array, parameters:np.array, intersection:float) -> np.array:
        if len(parameters)!= self.obs:
            raise Exception (f"parameters={parameters} have the wrong size")
        
        if parameters.shape[1] != input.shape[0]:
            raise Exception (f"Wrong parameter <=> input shape, {parameters.shape[1]}!={input.shape[0]}")

        if intersection.shape[0]!=input.shape[0]:
            raise Exception (f"Wrong intersection <=> input shape, {intersection.shape[0]}!={input.shape[0]}")

        # This is an array, where each column represents the i-th prediction
        # for input in x
        return np.dot(parameters,input)+intersection
    
    def _Cost(self, parameters:np.array, intersection:float) -> float:
        
        predictions = self.Predict(self.x, parameters,intersection)
        
        if predictions.shape!=self.y.shape:
            raise Exception (f"targets.shape({self.y.shape})!=predictions.shape({predictions.shape})")

        return (0.5/self.obs)*np.sum((predictions-self.y)**2)

    def _Derivatives(self, parameters, intersection) -> Tuple[np.array,float]:
        
        predictions = self.Predict(self.x, parameters, intersection)
        errors = (predictions-self.y)**2

        param_der = (1/self.obs)*np.dot(errors,self.x.T)
        b_der = (1/self.obs)*np.sum(errors)

        # Check derivatives has the correct dimensions
        if param_der.shape!=(self.feat,1):
            raise Exception ("Something went wrong, derivatives has incorrect size!")
        
        return param_der,b_der

    def GradDesc(self,alpha:float) -> Tuple[np.array, float, np.array]:
        
        alphas = [0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0][::-1]

        for alpha in alphas:
            w = np.ones(self.feat)
            b = 1.0
            k = alpha/self.obs
            prev_cost = self._Cost(w,b)
            finalized=True
            log_costs = np.array([])
            for i in range(self.iter+1):
                # Start gradient descent with generic values (1s for instance)
                w_dev,b_dev = self._derivatives(w,b)
                w -= k*w_dev
                b -= k*b_dev

                # Compute cost for each w and b
                cost = self._Cost(w,b)

                if cost>prev_cost:
                    warnings.warn("Cost is increasing, alpha is too big")
                    finalized=False
                    break
                prev_cost = cost
                log_costs.append(cost)
                print(f"Cost={cost}, iteration={i}")
            
            # Since we are starting with the highest alpha, the first alpha
            # that makes gradient descent converge is the most optimal one,
            # so we don't need to iterate over the rest
            if finalized:
                print("Gradient Descent has finished")
                return (w,b,log_costs)
        # if we get here is because none of the alphas made gradient descent to converge
        # in which case raise a program error
        raise ValueError(f"None of the alphas={self.alphas} makes Gradient Descent converge, check")