"""
Library containing multiple unsupervised algorithms
"""

import numpy as np
from typing import List, Sequence, Dict
import matplotlib.pyplot as plt
import warnings

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