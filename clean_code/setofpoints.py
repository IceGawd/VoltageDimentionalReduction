import numpy as np
from typing import Tuple, Optional

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
        
        if points.ndim != 2:
            raise ValueError("Points array must be 2-dimensional (n, d).")

        # Create weights if not given
        if weights is None:
            weights = np.ones(points.shape[0])

        weights /= np.sum(weights)        
        
        if weights.ndim != 1:
            raise ValueError("Weights array must be 1-dimensional (n,).")
        if points.shape[0] != weights.shape[0]:
            raise ValueError("Number of points and number of weights must be the same.")

        self.points = points
        self.shape = points.shape
        self.weights = weights

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
        """
        total = np.sum(self.weights)
        if total == 0:
            raise ValueError("Total weight is zero. Cannot normalize.")
        self.weights = self.weights / total

    def subset(self, indices: np.ndarray) -> "SetOfPoints":
        """
        Returns a new SetOfPoints object containing only the selected indices.

        Args:
            indices (np.ndarray): An array of indices to include in the new subset.

        Returns:
            SetOfPoints: A new SetOfPoints object with selected points and weights.
        """
        return SetOfPoints(self.points[indices], self.weights[indices])

    def __len__(self) -> int:
        """
        Returns the number of points in the set.

        Returns:
            int: Number of points.
        """
        return self.points.shape[0]

    def dimension(self) -> int:
        """
        Returns the dimensionality of the points.

        Returns:
            int: The dimension (d) of each point.
        """
        return self.points.shape[1]