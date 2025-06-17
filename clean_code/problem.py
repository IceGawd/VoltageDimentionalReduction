import setofpoints
import landmark
import solver

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

	def __init__(self, points: setofpoints.SetOfPoints, r=1.0):
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

		self.Rmat= None  # Resistance matrix
		self.G = None   # Graph representation of the resistance network

	def calcResistanceMatrix(self, k: int = 10, universalGround: bool = True) -> np.ndarray:
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
		nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
		_, indices = nbrs.kneighbors(X)

		# Dense kernel (n × n)
		kernel = np.zeros((n, n), dtype=float)
		weight = 1.0 / k

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
		landmark: landmark.Landmark, 
		target_avg_voltage: float = 0.1, 
		accuracy: float = 0.1, 
		radius: int = 3, 
		max_iter: int = 30, 
		k: int = 10) -> tuple[float, np.ndarray]:
		"""
		Finds the value of r (ground resistance) that makes the average voltage within a given radius
		as close as possible to the target average voltage, using binary search. returns the best r found and the voltages.

		Args:
			landmark (Landmark): The landmark node.
			target_avg_voltage (float): The target average voltage in the neighborhood.
			accuracy (float): Relative accuracy for stopping criterion.
			radius (int): Number of graph hops within which to compute the average voltage.
			max_iter (int): Maximum number of binary search iterations.
		"""
		# Build resistance graph once (exclude ground node)
		if self.Rmat is None:
			self.Rmat = self.calcResistanceMatrix(k=k, universalGround=True)
			self.Graph = nx.from_numpy_array(self.Rmat[:-1, :-1])  # Exclude ground node for graph connectivity

		# Find all nodes within 'radius' hops of the landmark

		lengths = nx.single_source_shortest_path_length(self.Graph, landmark.index, cutoff=radius)
		indices=np.array(list(lengths.keys()))

		r_init=indices.shape[0] #initial guess for r based on the number of neighbors within radius radius
		left, right = r_init/10,r_init*10  # Initial search range for r
		best_r = None
		best_loss = float('inf')

		for i in range(max_iter):
			r_try = np.exp((np.log(left) + np.log(right)) / 2)
			self.r = r_try  # Update problem resistance
			print(i,r_try)

			volt_solver = solver.Solver(self)
			voltages = volt_solver.compute_voltages(landmark, k=k)
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
		return best_r, voltages
	