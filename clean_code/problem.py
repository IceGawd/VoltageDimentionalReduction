import setofpoints

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import lil_matrix, csr_matrix, hstack, vstack
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

    def calcResistanceMatrix(self, k: int = 10, sparse: bool = True) -> Union[np.ndarray, csr_matrix]:
        """
        Calculates the (n+1)x(n+1) resistance matrix using k-nearest neighbors.
    
        Args:
            k (int): Number of nearest neighbors for sparse approximation.
            sparse (bool): Whether to return a sparse matrix.
    
        Returns:
            Union[np.ndarray, csr_matrix]: (n+1)x(n+1) resistance matrix.
        """
        X = self.points.points  # Assumes SetOfPoints instance
        n = X.shape[0]
    
        # Use NearestNeighbors to find k neighbors (including self, but we will exclude self manually)
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)
    
        # Build the resistance (affinity) matrix as sparse
        kernel = lil_matrix((n, n))
    
        for i in range(n):
            for j_idx, dist in zip(indices[i][1:], distances[i][1:]):  # skip self (first entry)
                # Unweighted NN
                weight = 1.0 / k
                # Weighted within nearest neighbor
                # weight = np.exp(-dist ** 2 / (self.c ** 2))
                kernel[i, j_idx] = weight
                kernel[j_idx, i] = weight  # symmetric
    
        # Ground node connection
        ground_column = np.full((n, 1), 1.0 / self.r)
        ground_row = ground_column.T
    
        # Convert kernel to CSR and build full matrix
        kernel = kernel.tocsr()
        
        if sparse:
            # Build (n+1)x(n+1) matrix
            last_row = csr_matrix(np.append(ground_row, 1.0 / self.r))
            full_matrix = vstack([
                hstack([kernel, csr_matrix(ground_column)]),
                last_row
            ])
            return full_matrix
    
        else:
            dense_kernel = kernel.toarray()
            kernel_with_ground = np.block([
                [dense_kernel,     ground_column],
                [ground_row, np.array([[1.0 / self.r]])]
            ])
            return kernel_with_ground


        
