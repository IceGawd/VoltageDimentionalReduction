"""
Provides the SetOfPoints class for managing weighted point sets in d-dimensional space.

This module implements functionality for handling collections of points with associated 
weights, supporting operations like normalization, subsetting, and point access. It's 
particularly useful for geometric algorithms and weighted data representations.

Example:
    >>> points = np.array([[1, 2], [3, 4], [5, 6]])
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> point_set = SetOfPoints(points, weights)
    >>> point_set.normalize_weights()
"""

import numpy as np
from typing import Tuple, Optional, Union, List

class SetOfPoints:
	"""
	Represents a set of points in a d-dimensional space along with associated weights.

	Attributes:
		points (np.ndarray): A 2D numpy array of shape (n, d), where each row is a point in d-dimensional space.
		weights (np.ndarray): A 1D numpy array of shape (n,) representing the weight for each point.
	"""

	def __init__(self, points: np.ndarray, weights: Optional[np.ndarray] = None):
		"""
		Initializes a SetOfPoints instance.

		Args:
			points (np.ndarray): A (n, d) array of n points in d-dimensional space.
			weights (Optional[np.ndarray]): A (n,) array of weights corresponding to the points.

		Raises:
			ValueError: If points and weights have incompatible shapes.
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

	def get_point(self, index: int) -> Tuple[np.ndarray, float]:
		"""
		Returns a specific point and its weight.

		Args:
			index (int): Index of the point to retrieve.

		Returns:
			Tuple[np.ndarray, float]: A tuple containing the point (1D array) and its weight.
		"""
		return self.points[index], self.weights[index]

	def normalize_weights(self) -> None:
		"""
		Normalizes the weights so that they sum to 1.

		Modifies the weights in-place, scaling them so their sum equals 1.
		This is useful for ensuring the weights represent a probability distribution.

		Raises:
			ValueError: If the sum of weights is zero.

		Example:
			>>> points = SetOfPoints(np.array([[1,2]]), weights=np.array([2.0]))
			>>> points.normalize_weights()
			>>> assert np.isclose(points.weights.sum(), 1.0)
		"""
		total = np.sum(self.weights)
		if total == 0:
			raise ValueError("Total weight is zero. Cannot normalize.")
		self.weights = self.weights / total

	def subset(self, indices: np.ndarray) -> "SetOfPoints":
		"""
		Returns a new SetOfPoints object containing only the selected indices.

		Creates a new instance with a subset of the original points and their 
		corresponding weights. The weights are preserved but not re-normalized.

		Args:
			indices (np.ndarray): An array of indices to include in the new subset.
				Can be boolean mask or integer indices.

		Returns:
			SetOfPoints: A new SetOfPoints object with selected points and weights.

		Example:
			>>> points = SetOfPoints(np.array([[1,2], [3,4], [5,6]]))
			>>> subset = points.subset(np.array([0, 2]))
			>>> print(len(subset))
			2
		"""
		return SetOfPoints(self.points[indices], self.weights[indices])

	def __len__(self) -> int:
		"""
		Returns the number of points in the set.

		Returns:
			int: Number of points.
		"""
		return self.points.shape[0]

	def __getitem__(self, index: Union[int, List[int], np.ndarray]) -> np.ndarray:
		"""
		Allows indexing into the dataset using array-like syntax.

		Supports integer indexing, slicing, and boolean masking to access points.
		Does not return weights - use get_point() if you need both point and weight.

		Args:
			index (Union[int, List[int], np.ndarray]): Index, slice, or boolean mask
				to select points.

		Returns:
			np.ndarray: The selected data point(s).

		Example:
			>>> points = SetOfPoints(np.array([[1,2], [3,4]]))
			>>> point = points[0]  # Get first point
			>>> subset = points[np.array([True, False])]  # Boolean indexing
		"""
		return self.points[index]

	def __setitem__(self, index: Union[int, List[int], np.ndarray], value: np.ndarray) -> None:
		"""
		Sets values in the dataset using array-like syntax.

		Supports integer indexing, slicing, and boolean masking to modify points.
		Only modifies points, not weights.

		Args:
			index (Union[int, List[int], np.ndarray]): Index, slice, or boolean mask
				indicating which points to modify.
			value (np.ndarray): The new value(s) to set. Must have compatible shape
				with the indexed points.

		Example:
			>>> points = SetOfPoints(np.array([[1,2], [3,4]]))
			>>> points[0] = np.array([5,6])  # Modify first point
		"""
		self.points[index] = value

	def dimension(self) -> int:
		"""
		Returns the dimensionality of the points in the set.

		Returns:
			int: The dimension (d) of each point in the space.

		Example:
			>>> points = SetOfPoints(np.array([[1, 2, 3], [4, 5, 6]]))
			>>> print(points.dimension())
			3
		"""
		return self.points.shape[1]