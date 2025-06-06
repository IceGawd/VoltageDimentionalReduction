import setofpoints

import numpy as np
from scipy.spatial.distance import cdist
from typing import Union, Optional
from sklearn.neighbors import NearestNeighbors

class Problem:
	"""
	Represents a kernel-based resistance model over a set of points with grounding.

	Attributes:
		points (SetOfPoints): The points object.
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

	def calcResistanceMatrix(self, k: int = 10) -> np.ndarray:
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
				kernel[i, j] += weight
				kernel[j, i] += weight					# keep it symmetric

		# Constant connection to the ground node
		connectivity = 1.0 / self.r						# self.r must be defined elsewhere
		ground_col = np.full((n, 1), connectivity, dtype=float)
		ground_row = ground_col.T						# (1 × n)

		# Assemble full (n+1) × (n+1) matrix
		top    = np.hstack((kernel, ground_col))		# (n × (n+1))
		bottom = np.hstack((ground_row, [[connectivity]]))
		full   = np.vstack((top, bottom))				# ((n+1) × (n+1))

		# Normalize so each row sums to 0 with diagonals 1
		row_sums = full.sum(axis=1, keepdims=True)
		weights = full / row_sums
		return np.identity(n + 1) - weights