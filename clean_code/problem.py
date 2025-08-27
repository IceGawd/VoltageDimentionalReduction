"""
Electrical Network Problem Definition for Dimensionality Reduction

This module implements an electrical network model for dimensionality reduction
using resistance networks. It creates a graph where nodes represent data points
and edges represent electrical connections, with a ground node added for reference.

The network is constructed using k-nearest neighbors to create sparse connections,
and the resistance matrix is computed to model the electrical properties of the network.

Example:
	>>> from problem import Problem
	>>> from setofpoints import SetOfPoints
	>>> points = SetOfPoints(data, weights)
	>>> problem = Problem(points, k=10, r=1.0)
	>>> resistance_matrix = problem.getResistanceMatrix()
"""

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
	Represents a kernel-based resistance network model over a set of centroids with grounding.

	This class implements an electrical network where:
	- Each centroid is a node in the network
	- Connections between nodes are weighted by their kernel values
	- A ground node is added with uniform resistance to all points
	- The network is sparsified using k-nearest neighbors

	Attributes:
		centroids (SetOfPoints): Collection of points and their weights that form
			the nodes of the network.
		r (float): Resistance to ground, controls the strength of connection to
			the reference ground node.
		ResistanceMatrix (np.ndarray): The computed (n+1)x(n+1) resistance matrix,
			where n is the number of centroids. The last row/column corresponds
			to the ground node.

	Note:
		- The resistance matrix is computed during initialization
		- The network is sparse, using only k-nearest neighbors
		- All resistances are normalized to create a proper probability matrix
		- See documentation link for detailed mathematical formulation
	"""

	def __init__(self, centroids: setofpoints.SetOfPoints, k: int = 10, r: float = 1.0) -> None:
		"""
		Initialize a new Problem instance with the given centroids and parameters.

		Creates an electrical network model where centroids are nodes connected
		through weighted edges, with an additional ground node. The network is
		sparsified by only connecting each node to its k nearest neighbors.

		Args:
			centroids (SetOfPoints): Collection of points that will form the network
				nodes. Each point should have an associated weight.
			k (int, optional): Number of nearest neighbors to connect each point to.
				Controls the sparsity of the network. Defaults to 10.
			r (float, optional): Resistance to the ground node. Higher values mean
				weaker connection to ground. Defaults to 1.0.

		Raises:
			ValueError: If ground resistance r is not positive.
			TypeError: If centroids is not a SetOfPoints instance.
			ValueError: If k is larger than the number of points.

		Example:
			>>> points = SetOfPoints(data, weights)
			>>> problem = Problem(points, k=15, r=2.0)
		"""
		if r <= 0:
			raise ValueError("Ground resistance (r) must be positive.")

		self.centroids = centroids
		self.r = r
		self.ResistanceMatrix =  self.calcResistanceMatrix(k,r)

	def calcResistanceMatrix(self, k: int = 10, r: float = 1.0) -> np.ndarray:
		"""
		Calculate the resistance matrix for the electrical network model.

		Constructs a sparse resistance matrix using k-nearest neighbors approach:
		1. Finds k nearest neighbors for each point
		2. Creates weighted connections between neighbors
		3. Adds ground node connections
		4. Normalizes to create probability matrix
		5. Converts to resistance matrix format

		For detailed mathematical formulation, see:
		https://github.com/IceGawd/VoltageDimentionalReduction/blob/main/highLevelDocs/VoltageCalculation.md

		Args:
			k (int, optional): Number of nearest neighbors for sparse approximation.
				Larger k means denser connections but slower computation. Defaults to 10.
			r (float, optional): Resistance to ground node. Controls the strength of
				the reference connection. Defaults to 1.0.

		Returns:
			np.ndarray: (n+1)x(n+1) resistance matrix where:
				- First n rows/columns correspond to centroids
				- Last row/column corresponds to ground node
				- Matrix is symmetric and row-normalized
				- Diagonal elements are 1

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

	def getResistanceMatrix(self) -> np.ndarray:
		"""
		Retrieve the pre-computed resistance matrix.

		Returns:
			np.ndarray: The (n+1)x(n+1) resistance matrix computed during
			initialization, where n is the number of centroids.
		"""
		return self.ResistanceMatrix
