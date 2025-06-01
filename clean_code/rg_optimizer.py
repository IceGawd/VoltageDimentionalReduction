"""
rg_optimizer.py

Module for optimizing the ground resistance parameter r in the Problem class.

This module provides a function to search for the best value of r by minimizing a user-specified loss function (e.g., validation error, voltage error, etc.).
"""

import numpy as np
from typing import Callable, Any

class rg_optimizer:
    """
    Class for optimizing the ground resistance parameter r in a Problem instance.

    Attributes:
        problem (Problem): The Problem instance to optimize.
    """
    def __init__(self, problem):
        """
        Initializes the optimizer with a Problem instance.

        Args:
            problem (Problem): An instance of the Problem class.
        """
        self.problem = problem

    def optimize(self, landmark, target_avg_voltage=0.1, accuracy=0.1, radius=3, r_min=1e-6, r_max=1e6, max_iter=30):
        """
        Finds the value of r (ground resistance) that makes the average voltage within a given radius of the landmark
        as close as possible to the target average voltage, using binary search.

        Args:
            landmark (Landmark): The landmark (with index and voltage) to use as the center.
            target_avg_voltage (float): The target average voltage in the neighborhood.
            accuracy (float): Relative accuracy for stopping criterion.
            radius (float): The radius within which to compute the average voltage.
            r_min (float): Minimum r value to consider.
            r_max (float): Maximum r value to consider.
            max_iter (int): Maximum number of binary search iterations.
        Returns:
            best_r: The r value that achieves the desired accuracy (or closest found).
            best_loss: The corresponding loss value.
        """
        import numpy as np
        import networkx as nx

        # Use the resistance graph from the problem
        G = nx.from_scipy_sparse_matrix(self.problem.graph)
        # Find all nodes within 'radius' hops from the landmark
        lengths = nx.single_source_shortest_path_length(G, landmark.index, cutoff=radius)
        indices = np.array(list(lengths.keys()))

        def compute_avg_voltage(r):
            self.problem.r = r
            v = self.problem.calcVoltage(landmark, r)
            return np.mean(v[indices])

        best_r = None
        best_loss = float('inf')
        left, right = r_min, r_max
        for _ in range(max_iter):
            r = np.exp((np.log(left) + np.log(right)) / 2)
            avg_voltage = compute_avg_voltage(r)
            rel_error = abs(avg_voltage - target_avg_voltage) / abs(target_avg_voltage)
            if rel_error < best_loss:
                best_loss = rel_error
                best_r = r
            if rel_error < accuracy:
                break
            if avg_voltage > target_avg_voltage:
                left = r
            else:
                right = r
        return best_r, best_loss



