"""
Library containing multiple unsupervised algorithms
"""

import numpy as np
from typing import List, Sequence, Dict

class K_means:

    def __init__(self, x:Sequence, k:int, iters:int) -> None:
        # data points
        self.x = x
        # size of data set
        self.m = len(x)
        # clusters
        self.k = k
        if self.k<=0:
            raise ValueError(f"Number of clusters {self.k} has to be positive")
        elif self.k > self.m:
            raise ValueError(f"User can not specify more clusters ({self.k}) than data points ({self.m})")
        # position of centroids
        self.centroids = []
        # number of iterations
        self.iters = iters

    def _compute_centroids(self, iter:int) -> List[List[float]]:
        # The idea is to pick k random values as initial values
        # for the centroids
        if iter==0:
            random_idxs = np.random.random_integers(low=0, high=self.m-1, size=self.k)
            centroids = [self.x[idx] for idx in random_idxs]

        else:  
            # TO-DO: Implement centroids calculation when they are not initialized
            # If we are not on the first iter, we need to compute the centroid as the mean
            # position of all data points assigned to that cluster. We first need to assign
            # points to clusters




        return centroids
    
    def _compute_distance(point:np.array, centroid:np.array) -> float:
        if point.shape[1] != centroid.shape[1]:
            raise ValueError(f"data point size {point.shape[1]} does not match centroid size ({centroid.shape[1]})")

        return sum((point-centroid)**2)
    
    def _compute_cost(self, c_coord, c_points) -> float:
        cost = 0
        for c_label, points in c_points.items():
            cost += sum(self._compute_distance(self.x[points,:],c_coord[c_label]))
        return cost/self.m
    
    def _assign_points_to_centroids(self, centroids:List[List[float]]) -> Dict[int,List[int]]:
        c_points = dict((c_idx,[]) for c_idx in range(len(centroids)))
        for idx in range(self.m):
            min_dis = float("inf")
            min_idx = 0
            for c_idx,c_val in enumerate(centroids):
                c_dis = self._compute_distance(self.x[idx], c_val)
                if c_dis < min_dis:
                    min_dis = c_dis
                    min_idx = c_idx
            
            c_points[min_idx].append(idx)
        
        return c_points
    
    def compute_clusters(self) -> np.array:
        for rand_it in range(self.iters):
            i = 0
            cost = float("inf")
            min_cost_log = [cost]
            cost_log = []
            while i==0 or cost!=prev_cost:
                prev_cost = cost    
                # 1) Compute centroids
                centroids = self._compute_centroids(i)
                
                # 2) Assign points to centroids
                #    For each cluster centroid, we should have the list of indexes
                #    of x that are on that centroid
                centroids_points_map = self._assign_points_to_centroids(centroids)
                                
                # 3) Compute cost function
                cost = self._compute_cost(centroids,centroids_points_map)
                cost_log.append(cost)
                i += 1
            
            if cost_log[-1] < min_cost_log[-1]:
                # Finally return the centroids that have achieved the overall lowest
                # cost function
                min_cost_log = cost_log
                min_centroids_map = centroids_points_map

        return min_cost_log, min_centroids_map
