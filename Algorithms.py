# The whole idea of this package is to create multiple classes, where
# each class will be a different Machine Learning algorithm

# Exercise 1. Implement the class gradient descent
import numpy as np
from typing import Tuple
import warnings
import math


class GradDescLinReg():

    # TO-DO: Make gradient descent a standalone algorithm that given the predictions
    # alpha, derivatives and the number of iterations, returns the best fitted estimates

    # TO-DO: Adjust logging with the correct name of the class

    # TO-DO: Add regularization term to both linear regression and gradient descend
    # make sure that we are not regularizing b (intercept), not the end of the world if we do
    # though, maybe add an option to do that
    

    def __init__(self, x:np.array, y:np.array, iter:int, intercept:bool=False, lambda_:int=0.0) -> None:
        # Here x and y are the features and targets of the Training Set!
        self.x = x
        self.y = y
        self.obs,self.feat = x.shape
        self.iter = iter
        self.b = intercept
        # lambda is the intercept term
        self.lambda_ = lambda_

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

        reg_cost = (0.5*self.lambda_/self.obs)*np.sum(parameters**2)

        return (0.5/self.obs)*np.sum(errors_sq) + reg_cost

    def _Derivatives(self, parameters) -> np.array:
        
        predictions = self.Predict(self.x, parameters)
        errors = predictions-self.y
        
        # reg term is a 1xn array (n,1) in numpy
        # x.T is a nxm and errrors is a mx1
        reg_term = self.lambda_*parameters
        param_der = np.dot(self.x.T,errors) + reg_term
        
        # in case there's an intercept do not regularize that derivative
        if self.b:
            param_der[0] -= reg_term[0]

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
            prev_cost = self._Cost(w)
            finalized=True
            log_costs = list()
            for i in range(self.iter):
                
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
    

class GradDescLogReg:

    def __init__(self, x:np.array, y:np.array, iterations:int, alpha:float=0.0, intercept:bool=False, lambda_:int=0.0):
        self.x = x
        self.y = y
        self.obs,self.feat = x.shape
        self.iters = iterations
        self.b = intercept
        # lambda is the regularization parameter (when lambda =0, no regularization is applied)
        self.lambda_ = lambda_
        self.alpha = alpha

    def _Sigmoid(self,z:np.array):
        return 1.0/(1+np.exp(-1.0*z))
    
    def Predict(self, input:np.array, params:np.array) -> np.array:
        z = np.dot(input,params)
        return self._Sigmoid(z)
    
    def _SingleCost(self,x_i:np.array ,y_i:int, params:np.array) -> float:
        # This function computes the cost for an individual observation

        # Make sure that the label to be classified is binary (0,1)
        if y_i!=0 and y_i!=1:
            raise ValueError(f"target value, {y_i} is not binary, check")
        
        predict_val = self.Predict(x_i,params)
        ind_cost = -1.0*y_i*np.log(predict_val)-(1-y_i)*np.log(1-predict_val)
        return ind_cost
    
    def _Cost(self,params:np.array)-> float:
        # This function computes the overall cost for the params chosen
        # over the entire dataset (mean of individual costs for all observations)
        cost = 0.0
        for i in range(self.obs):
            cost += self._SingleCost(self.x[i],self.y[i],params)

        # Add regularization cost in case user wants it
        reg_cost = 0
        for j in range(len(params)):
            reg_cost +=  params[j]**2
        
        reg_cost = (self.lambda_/(2*self.obs))*reg_cost

        # cost should be a scalar at all times
        if type(cost)!=np.float_:
            raise ValueError(f"GradDescLogReg, cost is not a scalar, type={type(cost)}, check!")
        
        return (1/self.obs)*cost + reg_cost
    
    # Now the only thing missing is applying gradient descent
    # to the estimates. For that we need the derivatives
    # over the multiple terms
    def _Derivatives(self, parameters) -> np.array:
        
        predictions = self.Predict(self.x, parameters)
        errors = predictions-self.y
        
        # x is mxn, x.T is nxm and errors is mx1, lambda is a scalar that should mulitplya nx1, which is parameters
        param_der = np.dot(self.x.T,errors) + self.lambda_*parameters
        
        if self.b:
            # Normally we do not regularize the intercept term, so undo that if needed
            param_der[0] -= self.lambda_*parameters[0]

        return param_der
    
    def Compute_Parameters(self) -> Tuple[np.array,list]:
        
        if self.alpha!=0.0:
            alphas = [self.alpha]
        else:
            alphas = [0.0001,0.0003,0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0][::-1]

        for alpha in alphas:
            # Start gradient descent with generic values (0s for instance)
            w = np.zeros(self.feat)
            k = alpha/self.obs
            prev_cost = self._Cost(w)
            finalized = True
            log_costs = list()
            for i in range(self.iters):
                
                # Compute gradient and update parameters
                w_dev = self._Derivatives(w)
                w -= k*w_dev

                # Compute cost for each w (note the independent term is the first value in here)
                cost = self._Cost(w)

                if cost>prev_cost:
                    warnings.warn("LogRegression: Cost is increasing, alpha is too big")
                    finalized=False
                    break
                
                if i% math.ceil(self.iters / 10) == 0:
                    print(f"Iteration {i:4d}: Cost {cost}, w {w}") 
                prev_cost = cost
                log_costs.append(cost)
            
            # Since we are starting with the highest alpha, the first alpha
            # that makes gradient descent converge is the most optimal one,
            # so we don't need to iterate over the rest
            if finalized:
                print(f"LogRegression: Gradient Descent has finished for alpha={alpha}")
                return (w,log_costs)
        
        # if we get here is because none of the alphas made gradient descent to converge
        # in which case raise a program error
        raise ValueError(f"LogRegression: None of the alphas={alphas} makes /n" 
                         f" Gradient Descent converge, check")