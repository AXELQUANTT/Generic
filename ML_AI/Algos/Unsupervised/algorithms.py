"""
Library containing multiple unsupervised algorithms
"""

import numpy as np
from typing import List, Sequence, Dict, Union, Optional
import matplotlib.pyplot as plt
import warnings
import scipy.special
import scipy.stats as stats

class Anomaly_Detection:
    """
    Given a dataset containing a set of features, this algo
    is devoted to identify which of those data points are
    'anomalies'.

    This algorithm should be run:
    ad = Anomaly_Detection(x_test)
    epsilon = ad.compute_epsilon(x_cv, y_cv)
    anomalous = compute_anomalous(x_train,epsilon)

    The algorithm assumes that the features are
    continous random variables
    """ 

    def __init__(self, x:np.array) -> None:
        # Assumes number of anomalies in x is ~ 0 wrt len(x)
        self.x = x
        self.size,self.feat = x.shape
        self.dist = dict((feat_idx,()) for feat_idx in range(self.feat))
    
    def _get_distribution(self) -> None:
        """
        Main function of the algo devoted
        to compute the probability distributions
        of the input data
        """

        
        scipy.special.seterr(all='raise')
        #warnings.filterwarnings('error')
        # Get all continous probability distributions in scipy
        all_dist = [getattr(stats, d) for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_continuous)]
        for feat in range(self.feat):
            min_pval = float('inf')
            var = self.x[:,feat]
            for dist in all_dist:
                # First compute the meaningful parameters of the distribution
                try:
                    # Catch exceptions as it may be the case that the proposed distribution
                    # can not be fitted to the data
                    params = dist.fit(var)
                    fitt_dist = dist(*params)
                    tstat, pval = stats.ks_1samp(x=var, cdf=dist.cdf, args=params)
                    if pval == 0.0:
                        self.dist[feat] = (fitt_dist,pval)
                        break
                    elif pval < min_pval:
                        min_pval = pval
                        print(f'Dsitrbution to be fitted => {dist}')
                        self.dist[feat] = (fitt_dist,pval)
                except:
                    warnings.warn(f'Distribution {dist.name} can not be fitted to feature {feat}')
                    pass
    
    def _compute_prob(self, x:np.array) -> np.array:
        """
        From the previously computed distributions,
        calculate the probabiliy of occurrence of each 
        data point
        """
        
        probs = np.ones([len(x),])
        for feat in range(self.feat):
            prob_dist = self.dist[feat][0]
            probs = np.multiply(probs,prob_dist.pdf(x[:,feat]))
        return probs

    def compute_epsilon(self, x_ce:np.array, y_ce:np.array) -> float:
        """
        Main function of the class. It returns the epsilon
        that gets the higher rate of anomalous data points 
        correctly labeled. It uses binary search to achieve
        that
        """

        self._get_distribution()
        probabilities = self._compute_prob(x_ce)

        left = 0
        right = 1.0
        real_anomalies = sum(y_ce)
        while left <= right:
            epsilon = (left+right)/2
            detected_anomalies = sum(probabilities < epsilon)
            if detected_anomalies > real_anomalies:
                right =  epsilon - 0.01
            elif detected_anomalies < real_anomalies:
                left = epsilon + 0.01
            else:
                return epsilon
        return epsilon
    
    def compute_anomalous(self, x_new:np.array, epsilon:float) -> np.array:
        """
        Function return a np.ones[self.size,1] with 0s and 1s,
        where 1s indicate anomalous datapoints and 0 indicates
        regular data points
        """

        anomalous = np.array(self._compute_prob(x_new) < epsilon, dtype=int)
        anomalous.resize([len(x_new),1])
        return anomalous
        
class K_means:

    def __init__(self, x:Sequence, k:int, iters:int, max_iters:int) -> None:
        # data points
        self.x = x
        # size of dataset and number of features
        self.m, self.feat = x.shape
        # clusters
        self.k = k
        if self.k<=0:
            raise ValueError(f"Number of clusters, {self.k}, needs to be positive")
        elif self.k > self.m:
            raise ValueError(f"User can not specify more clusters ({self.k})"
                             f" than data points ({self.m})")
        # position of centroids
        self.centroids = [[]]
        # number of iterations
        self.iters = iters
        if self.iters <= 0:
            raise ValueError(f"number of iterations, {self.iters}, needs to be positive")
        # max number iterations within a single iter
        self.max_iters = max_iters
        if self.max_iters <=0:
            raise ValueError(f"number of iterations, {self.max_iters}, needs to be positive")
        

    def _compute_centroids(self, c_map:dict) -> List[List[float]]:
        # The idea is to pick k random values as initial values
        # for the centroids
        if not c_map:
            random_idxs = np.random.random_integers(low=0, high=self.m-1, size=self.k)
            centroids = np.array([self.x[idx] for idx in random_idxs])

        else:
            centroids = np.zeros([self.k,self.feat])
            idx = 0
            for values in c_map.values():
                centroids[idx,:] = np.mean(self.x[values], axis=0)
                idx += 1

        return centroids
    
    def _compute_distance(self, point:np.array, centroid:np.array) -> float:
        return sum((point-centroid)**2)
    
    def _compute_cost(self, c_coord, c_points) -> float:
        cost = 0
        for c_label, points in c_points.items():
            # Could be the case that no points are associated with a given centroid,
            # in which case we should stop the current iteration
            if self.x[points,:].size>0:
                cost += sum(self._compute_distance(self.x[points,:], c_coord[c_label]))
            else:
                warnings.warn(f"Cluster on coordinates {c_coord[c_label]} does not have"
                              f" any data point associated")
                return
        return cost/self.m
    
    def _assign_points_to_centroids(self, x, centroids:List[List[float]]) -> Dict[int,List[int]]:
        c_points = dict((c_idx,[]) for c_idx in range(len(centroids)))
        for idx in range(self.m):
            min_dis = float("inf")
            min_idx = 0
            for c_idx,c_val in enumerate(centroids):
                c_dis = self._compute_distance(x[idx], c_val)
                if c_dis < min_dis:
                    min_dis = c_dis
                    min_idx = c_idx
            
            c_points[min_idx].append(idx)
        
        return c_points
    
    def compute_clusters(self) -> np.array:
        min_cost_log = [float("inf")]
        min_centroids = [[]]
        for rand_it in range(self.iters):
            i = 0    
            cost = float("inf")
            cost_log = []
            centroids_points_map = {}
            while (not centroids_points_map or cost!=prev_cost) and i < self.max_iters:
                prev_cost = cost    
                # Compute centroids
                centroids = self._compute_centroids(centroids_points_map)
                
                # Assign points to centroids
                # For each cluster centroid, we should have the list of indexes
                # of x that are on that centroid
                centroids_points_map = self._assign_points_to_centroids(self.x, centroids)
                                
                # Compute cost function
                cost = self._compute_cost(centroids,centroids_points_map)
                if not cost:
                    warnings.warn("Cost can not be computed, stopping iteration")
                    cost_log.append(float("inf"))
                    break

                cost_log.append(cost)

                # Sanity check
                if cost>prev_cost:
                    raise ValueError(f"Cost increased after iteration, {i}, of tranche {rand_it},"
                                     f"with cost={cost} vs prev_cost={prev_cost}")
                i += 1
            
            if cost_log[-1] < min_cost_log[-1]:
                # Finally return the centroids that have achieved the overall lowest
                # cost function
                min_cost_log = cost_log
                min_centroids_map = centroids_points_map
                min_centroids = centroids

        self.centroids = min_centroids
        return min_cost_log, min_centroids ,min_centroids_map
    
    def predict(self, x_new:np.array) -> Dict[int,List[int]]:
        if self.centroids!=[[]]:
            return self._assign_points_to_centroids(x_new, self.centroids)
        else:
            raise ValueError("Centroids are not being computed yet,"
                             "run compute_clusters function first")