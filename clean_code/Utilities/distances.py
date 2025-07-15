"""
Distance computation utilities for point sets using various metrics.

This module provides functionality to compute distances between points using different
metrics: Euclidean distance, voltage-based distance, and graph-based distance.
It utilizes FAISS for efficient distance computations and scikit-learn for
k-nearest neighbors graph construction.

Dependencies:
    - numpy: For numerical computations
    - faiss: For efficient distance computations
    - scipy: For sparse matrix operations and graph algorithms
    - sklearn: For nearest neighbors computation
"""

import numpy as np
from scipy.sparse import csgraph

def compute_distances(point_set, voltages):
    """
    Compute three different types of distances between points.

    This function calculates three distance matrices using different metrics:
    1. Euclidean distance between points in the original space
    2. Euclidean distance between points in the voltage space
    3. Graph distance based on k-nearest neighbors connectivity

    Args:
        point_set (SetOfPoints): Set of points in the original space,
            containing the 'points' attribute as numpy array of shape (n_points, n_dimensions)
        voltages (np.ndarray): Voltage values for each point,
            array of shape (n_points, n_voltages)

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Three distance matrices:
            - D1: Euclidean distances in original space (n_points × n_points)
            - D2: Euclidean distances in voltage space (n_points × n_points)
            - D3: Graph-based distances (n_points × n_points)
        
        Note: Diagonal entries are set to -inf for D1 and D2, and graph
        unreachable nodes are set to inf in D3.

    Example:
        >>> points = SetOfPoints(np.random.rand(100, 2))
        >>> voltages = np.random.rand(100, 10)
        >>> D1, D2, D3 = compute_distances(points, voltages)
        >>> print(f"Distance matrices shapes: {D1.shape}, {D2.shape}, {D3.shape}")
    """
    # Import required libraries
    import faiss
    from scipy.sparse import lil_matrix
    from sklearn.neighbors import NearestNeighbors

    # Section 1: Compute Euclidean distances in original space
    X = point_set.points
    index = faiss.IndexFlatL2(X.shape[1])  # Create L2 distance index
    index.add(X.astype(np.float32))        # Add points to the index
    D1, _ = index.search(X.astype(np.float32), X.shape[0])
    np.fill_diagonal(D1, -np.inf)          # Set diagonal to -inf

    # Section 2: Compute distances based on voltages
    X = voltages
    index = faiss.IndexFlatL2(X.shape[1])  # Create L2 distance index
    index.add(X.astype(np.float32))        # Add points to the index
    D2, _ = index.search(X.astype(np.float32), X.shape[0])
    np.fill_diagonal(D2, -np.inf)          # Set diagonal to -inf

    # Section 3: Compute graph-based distances
    k = 5  # Get number of neighbors from config
    
    # Create k-nearest neighbors graph
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(point_set.points)
    _, indices = nbrs.kneighbors(point_set.points)
    
    # Create sparse adjacency matrix
    n = point_set.points.shape[0]
    adjacency_matrix = lil_matrix((n, n), dtype=np.float32)
    
    # Fill adjacency matrix with symmetric connections
    for i in range(n):
        for j in indices[i]:
            if i != j:
                adjacency_matrix[i, j] = 1.0
                adjacency_matrix[j, i] = 1.0
    
    # Compute shortest paths in the graph
    D3 = csgraph.shortest_path(adjacency_matrix, method="auto", directed=False, unweighted=True)
    D3[np.isinf(D3)] = np.inf  # Replace scipy's inf with numpy's inf for consistency

    return D1, D2, D3




