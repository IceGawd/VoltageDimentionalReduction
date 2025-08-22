"""
Weighted Point Set Management Module

This module provides a class for managing collections of points in d-dimensional
space, where each point has an associated weight. It supports operations like
normalization, subsetting, and point access while maintaining weight consistency.

The SetOfPoints class is particularly useful in dimensionality reduction and
clustering applications where point weights influence the importance of each
data point in the analysis.

Example:
	>>> import numpy as np
	>>> from setofpoints import SetOfPoints
	>>> # Create 100 2D points with equal weights
	>>> points = np.random.randn(100, 2)
	>>> point_set = SetOfPoints(points)
	>>> # Create weighted points
	>>> weights = np.random.uniform(0, 1, 100)
	>>> weighted_set = SetOfPoints(points, weights)
"""

import numpy as np
from typing import Tuple, Optional, Union, List
import numpy.typing as npt

class SetOfPoints:
	"""
	A collection of points in d-dimensional space with associated weights.

	This class manages a set of points where each point has an associated weight,
	maintaining consistency between points and weights, and providing operations
	for manipulation and access. Weights are automatically normalized to sum to 1.

	Attributes:
		points (np.ndarray): Points array of shape (n, d), where:
			- n is the number of points
			- d is the dimensionality of the space
			- points[i] is the i-th point's coordinates
		weights (np.ndarray): Weight array of shape (n,), where:
			- weights[i] is the normalized weight of points[i]
			- weights sum to 1.0
		shape (Tuple[int, int]): Shape tuple (n, d) matching points array

	Note:
		- All weights are normalized during initialization
		- Points must be 2D array even for 1D data (use shape (n, 1))
		- Weights must be non-negative and sum to a positive value
		- Supports numpy-style indexing and iteration
	"""

	def __init__(self, points: npt.NDArray, weights: Optional[npt.NDArray] = None) -> None:
		"""
		Initialize a new weighted point set.

		Creates a set of weighted points, normalizing the weights to sum to 1.
		If no weights are provided, uniform weights (1/n) are used.

		Args:
			points (np.ndarray): Array of points with shape (n, d) where:
				- n is the number of points
				- d is the dimensionality
				- Must be a 2D array even for 1D data
			weights (Optional[np.ndarray], optional): Array of weights with
				shape (n,). If not provided, uniform weights are used.
				Weights will be normalized to sum to 1. Defaults to None.

		Raises:
			TypeError: If points is not a numpy array
			ValueError: If any of:
				- points array is not 2D
				- weights array is not 1D
				- number of weights doesn't match number of points
				- weights sum to zero

		Example:
			>>> # 3 points in 2D with custom weights
			>>> pts = np.array([[0,0], [1,1], [2,2]])
			>>> wts = np.array([1.0, 2.0, 1.0])
			>>> point_set = SetOfPoints(pts, wts)
			>>> print(point_set.weights)  # Shows normalized weights
		"""
		
		if not isinstance(points, np.ndarray):
			raise TypeError("Points must be a 2Dnumpy array.")
		if points.ndim != 2:
			raise ValueError("Points array must be 2-dimensional (n, d).")

		# Create weights if not given
		if weights is None:
			weights = np.ones(points.shape[0])

		new_weights = weights / np.sum(weights)
		self.weights = new_weights

		if self.weights.ndim != 1:
			raise ValueError("Weights array must be 1-dimensional (n,).")
		if points.shape[0] != self.weights.shape[0]:
			raise ValueError("Number of points and number of weights must be the same.")

		self.points = points
		self.shape = points.shape

	def get_point(self, index: int) -> Tuple[npt.NDArray, float]:
		"""
		Retrieve a specific point and its associated weight.

		Provides convenient access to both the coordinates and weight
		of a point at the specified index.

		Args:
			index (int): Index of the point to retrieve. Must be in
				range [0, n-1] where n is the number of points.

		Returns:
			Tuple[np.ndarray, float]: Tuple containing:
				- Point coordinates as 1D array of shape (d,)
				- Associated normalized weight as float

		Raises:
			IndexError: If index is out of range

		Example:
			>>> point_set = SetOfPoints(points)
			>>> coords, weight = point_set.get_point(0)
			>>> print(f"Point: {coords}, Weight: {weight:.3f}")
		"""
		return self.points[index], self.weights[index]

	def normalize_weights(self) -> None:
		"""
		Normalize the weights to sum to 1.0.

		Adjusts all weights proportionally so their sum equals 1 while
		maintaining their relative ratios. This ensures the weights
		form a proper probability distribution over the points.

		Raises:
			ValueError: If all weights are zero (cannot normalize)

		Example:
			>>> points = np.array([[1,1], [2,2]])
			>>> weights = np.array([2.0, 3.0])
			>>> point_set = SetOfPoints(points, weights)
			>>> point_set.weights  # array([0.4, 0.6])

		Note:
			This method is called automatically during initialization,
			but can be used to re-normalize weights if they are
			modified after creation.
		"""
		total = np.sum(self.weights)
		if total == 0:
			raise ValueError("Total weight is zero. Cannot normalize.")
		self.weights = self.weights / total

	def subset(self, indices: npt.NDArray[np.int_]) -> "SetOfPoints":
		"""
		Create a new SetOfPoints containing only specified points.

		Creates a new instance containing only the points and weights at
		the specified indices. The weights in the new set are normalized
		to sum to 1.

		Args:
			indices (np.ndarray): Integer array of indices to include.
				Must be valid indices in range [0, n-1].

		Returns:
			SetOfPoints: New SetOfPoints instance containing:
				- Selected points at specified indices
				- Corresponding weights, renormalized to sum to 1

		Example:
			>>> point_set = SetOfPoints(points)  # 100 points
			>>> # Select first 10 points
			>>> subset = point_set.subset(np.arange(10))
			>>> print(len(subset))  # 10

		Note:
			The returned subset is a new instance with independent
			points and weights arrays.
		"""
		return SetOfPoints(self.points[indices], self.weights[indices])

	def __len__(self) -> int:
		"""
		Returns the number of points in the set.

		Returns:
			int: Number of points.
		"""
		return self.points.shape[0]

	def __getitem__(self, index):
		"""
		Allows indexing into the dataset.

		Args:
			index (int): Index of the desired data point.

		Returns:
			np.ndarray: The data point at the given index.
		"""
		return self.points[index]

	def __setitem__(self, index, value):
		"""
		Sets a value in the dataset at a specified index.

		Args:
			index (int): The index to modify.
			value (Any): The new value to set.
		"""
		self.points[index] = value

	def dimension(self) -> int:
		"""
		Get the dimensionality of the point space.

		Returns the number of dimensions (d) in which the points exist.
		This is the second component of the points array shape (n, d).

		Returns:
			int: Number of dimensions in the point space

		Example:
			>>> points = np.random.randn(100, 3)  # 100 3D points
			>>> point_set = SetOfPoints(points)
			>>> print(point_set.dimension())  # 3

		Note:
			Even for 1D data, points are stored as 2D array with
			shape (n, 1), so dimension() will return 1.
		"""
		return self.points.shape[1]