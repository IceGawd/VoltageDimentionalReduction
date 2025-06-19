import setofpoints
import landmark
import solver
import config

import numpy as np
from scipy.spatial.distance import cdist
from typing import Union, Optional, List
from sklearn.neighbors import NearestNeighbors
import networkx as nx

class Problem:
	"""
	Represents a kernel-based resistance model over a set of points with grounding.

	Attributes:
		points (SetOfPoints): The points object.
		landmarks (List[Landmark])
		c (float): Kernel width parameter used in the Gaussian kernel.
		r (float): Resistance to ground.
	"""

	def __init__(self, points: setofpoints.SetOfPoints, r: float):
		"""
		Initializes a Problem instance.

		Args:
			points (np.ndarray): A (n, d) array of points.
			r (float): Resistance to the ground.

		Raises:
			ValueError: If input dimensions are incorrect or parameters are non-positive.
		"""
		if r <= 0:
			raise ValueError("Ground resistance (r) must be positive.")

		self.points = points
		self.r = r

	def calcResistanceMatrix(self, universalGround: bool = True) -> np.ndarray:
		"""
		Calculates the (n+1)x(n+1) row-normalized resistance matrix using k-nearest neighbors.

		Args:
			k (int): Number of nearest neighbors for sparse approximation.
			sparse (bool): Whether to return a sparse matrix.

		Returns:
			np.ndarray: (n+1)x(n+1) resistance matrix with rows summing to 1.
		"""

		X = self.points.points							# shape (n, d)
		n = X.shape[0]

		# k-NN search (k+1 because the first neighbor is the point itself)
		nbrs = NearestNeighbors(n_neighbors=config.params['k-nearest-neighbors'] + 1, algorithm='auto').fit(X)
		_, indices = nbrs.kneighbors(X)

		# Dense kernel (n × n)
		kernel = np.zeros((n, n), dtype=float)
		weight = 1.0 / config.params['k-nearest-neighbors']

		for i in range(n):
			for j in indices[i][1:]:					# skip the point itself
				kernel[i, j] = weight * self.points.weights[i] * self.points.weights[j] 
				kernel[j, i] = weight * self.points.weights[j] * self.points.weights[i]	# keep it symmetric

		# Constant connection to the ground node
		if (universalGround):
			connectivity = kernel.sum() / (self.r * n * n)
			ground_col = np.full((n, 1), connectivity, dtype=float)
			ground_row = ground_col.T						# (1 × n)

			# Assemble full (n+1) × (n+1) matrix
			top    = np.hstack((kernel, ground_col))		# (n × (n+1))
			bottom = np.hstack((ground_row, [[0]]))
			full   = np.vstack((top, bottom))				# ((n+1) × (n+1))
		else:
			full = kernel

		# Normalize so each row sums to 0 with diagonals 1
		row_sums = full.sum(axis=1, keepdims=True)
		weights = full / row_sums
		return np.identity(weights.shape[0]) - weights

	def optimize(self, 
		landmarks: List[landmark.Landmark], 
		target_avg_voltage: float = 0.1, 
		accuracy: float = 0.01, 
		radius: int = 3, 
		r_min: float = 0.01, 
		r_max: float = 100, 
		max_iter: int = 30):
		"""
		Finds the value of r (ground resistance) that makes the average voltage within a given radius
		as close as possible to the target average voltage, using binary search.

		Args:
			target_avg_voltage (float): The target average voltage in the neighborhood.
			accuracy (float): Relative accuracy for stopping criterion.
			radius (int): Number of graph hops within which to compute the average voltage.
			r_min (float): Minimum r value to consider.
			r_max (float): Maximum r value to consider.
			max_iter (int): Maximum number of binary search iterations.
		"""
		# Build resistance graph once (exclude ground node)
		R = self.calcResistanceMatrix()
		G = nx.from_numpy_array(R[:-1, :-1])  # Exclude ground node for graph connectivity

		# Find all nodes within 'radius' hops of the landmark
		indices = []
		for landmark in landmarks:
			lengths = nx.single_source_shortest_path_length(G, landmark.index, cutoff=radius)
			indices.extend(list(lengths.keys()))

		indices = np.array(indices)

		best_r = None
		best_loss = float('inf')
		left, right = r_min, r_max

		for i in range(max_iter):
			r_try = np.exp((np.log(left) + np.log(right)) / 2)
			self.r = r_try  # Update problem resistance
			# print(r_try)

			volt_solver = solver.Solver(self)
			voltages = volt_solver.compute_voltages(landmarks)
			neighborhood_avg = np.mean(voltages[indices])
			rel_error = abs(neighborhood_avg - target_avg_voltage)

			# print(neighborhood_avg)

			if rel_error < best_loss:
				best_loss = rel_error
				best_r = r_try

			if rel_error < accuracy:
				break

			if neighborhood_avg < target_avg_voltage:
				left = r_try
			else:
				right = r_try

		self.r = best_r