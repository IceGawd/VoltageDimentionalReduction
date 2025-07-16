import setofpoints
import landmark
import solver
from Utilities import config

import numpy as np
from scipy.spatial.distance import cdist
from typing import Union, Optional, List
from sklearn.neighbors import NearestNeighbors
import networkx as nx

class Problem:
	"""
	Represents a kernel-based resistance model over a set of centroids with grounding.

	Attributes:
		centroids (SetOfPoints): The centroids object.
		landmarks (List[Landmark])
		c (float): Kernel width parameter used in the Gaussian kernel.
		r (float): Resistance to ground.
	"""

	def __init__(self, centroids: setofpoints.SetOfPoints, k: int = 10, r: float = 1.0):
		"""
		Initializes a Problem instance.

		Args:
			centroids: A SetOfPoints: stores points and weights.
			r (float): Resistance to the ground.

		Raises:
			ValueError: If input dimensions are incorrect or parameters are non-positive.
		"""
		if r <= 0:
			raise ValueError("Ground resistance (r) must be positive.")

		self.centroids = centroids
		self.r = r
		self.ResistanceMatrix =  self.calcResistanceMatrix(k,r)

	def calcResistanceMatrix(self, k: int = 10,r: float = 1.0) -> np.ndarray:
		"""
		Calculates the (n+1)x(n+1) row-normalized resistance matrix using k-nearest neighbors.
		See for explaination: https://github.com/IceGawd/VoltageDimentionalReduction/blob/main/highLevelDocs/VoltageCalculation.md

		Args:
			k (int): Number of nearest neighbors for sparse approximation.
			r (float): Resistance to ground

		Returns:
			np.ndarray: (n+1)x(n+1) resistance matrix with rows summing to 1.
		"""

		print(type(self.centroids))
		X = self.centroids.points						# shape (n, d)
		n = X.shape[0]

		# k-NN search (k+1 to cover self-inclusion)
		nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
		_, indices = nbrs.kneighbors(X)

		# Dense kernel (n × n)
		kernel = np.zeros((n, n), dtype=float)
		weight = 1.0 / k

		for i in range(n):
			for j in indices[i]:
				if j != i:
					kernel[i, j] = weight * self.centroids.weights[i] * self.centroids.weights[j]
					kernel[j, i] = weight * self.centroids.weights[j] * self.centroids.weights[i]  # symmetric


		kernel = kernel / kernel.sum(axis=1, keepdims=True)

		# Constant connection to the ground node
		connectivity = 1 / self.r
		ground_col = np.full((n, 1), connectivity, dtype=float)
		ground_row = ground_col.T						# (1 × n)

		# Assemble full (n+1) × (n+1) matrix
		top    = np.hstack((kernel, ground_col))		# (n × (n+1))
		bottom = np.hstack((ground_row, [[0]]))
		full   = np.vstack((top, bottom))				# ((n+1) × (n+1))

		# Normalize so each row sums to 0 with diagonals 1
		row_sums = full.sum(axis=1, keepdims=True)
		probabilties = full / row_sums
		return np.identity(probabilties.shape[0]) - probabilties

	def getResistanceMatrix(self):
		return self.ResistanceMatrix
