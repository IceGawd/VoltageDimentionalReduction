import setofpoints

import numpy as np
from typing import Optional, Callable

class KMeans:
    """
    Streaming KMeans clustering algorithm that creates a weighted partition
    (SetOfPoints) from a data file or large dataset.

    This is suitable for large-scale or streaming data, where full dataset
    access is not feasible.
    """

    def __init__(self, k: int):
        """
        Initializes the streaming KMeans.

        Args:
            k (int): Number of clusters (partitions).
            distance_fn (Callable): Optional custom distance function.
                                    Defaults to Euclidean distance.
        """
        self.k = k
        self.centers = None
        self.weights = None
        self.counts = None

    def fit_on_set(self, set_of_points: setofpoints.SetOfPoints) -> setofpoints.SetOfPoints:
        """
        Fit KMeans directly on a SetOfPoints object by streaming over weighted points.
    
        Args:
            set_of_points (SetOfPoints): Input set to cluster.

        Returns:
            SetOfPoints: The clustered centers with updated weights.
        """
        points, weights = set_of_points.points, set_of_points.weights

        i = 0
        for point, weight in zip(points, weights):
            if (int(i * 100 / len(points)) != int((i - 1) * 100 / len(points))):
                print(str(int(i * 100 / len(points))) + "%")
            self._update_weighted(point, weight)
            i += 1
    
        return SetOfPoints(points=self.centers, weights=self.weights)

    def _update_weighted(self, point: np.ndarray, weight: float):
        """
        Updates the cluster centers with a weighted point.
    
        Args:
            point (np.ndarray): A single data point.
            weight (float): The weight of the data point.
        """
        if self.centers is None:
            self.centers = np.array([point])
            self.counts = np.array([weight])
            self.weights = np.array([weight])
            return
    
        if len(self.centers) < self.k:
            self.centers = np.vstack([self.centers, point])
            self.counts = np.append(self.counts, weight)
            self.weights = np.append(self.weights, weight)
            return
    
        # Find closest center
        dists = [np.linalg.norm(c - point) for c in self.centers]
        closest_index = np.argmin(dists)
    
        # Update center using weighted running mean
        total_weight = self.counts[closest_index] + weight
        eta = weight / total_weight
        self.centers[closest_index] = (1 - eta) * self.centers[closest_index] + eta * point
        self.counts[closest_index] = total_weight
        self.weights[closest_index] += weight