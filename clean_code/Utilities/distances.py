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

# Try to import required packages
try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False
    print("Warning: faiss not found, please install it for optimal performance")
    
try:
    from scipy.sparse import lil_matrix, csgraph
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("Warning: scipy not found, please install it for graph distances")
    
try:
    from sklearn.neighbors import NearestNeighbors
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False
    print("Warning: scikit-learn not found, please install it for graph distances")

def compute_distances(points, voltages):
    """
    Compute three different types of distances between points.

    This function calculates three distance matrices using different metrics:
    1. Euclidean distance between points in the original space
    2. Euclidean distance between points in the voltage space
    3. Graph distance based on k-nearest neighbors connectivity

    Args:
        points (Union[np.ndarray, SetOfPoints]): Points in the original space.
            Can be either a numpy array of shape (n_points, n_dimensions) or
            a SetOfPoints instance containing the points.
        voltages (Union[np.ndarray, VoltageMap]): Voltage values for each point.
            Can be either a numpy array of shape (n_points, n_voltages) or
            a VoltageMap instance containing the voltages.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Three distance matrices:
            - D1: Euclidean distances in original space (n_points × n_points)
            - D2: Euclidean distances in voltage space (n_points × n_points)
            - D3: Graph-based distances (n_points × n_points)
        
        Note: Diagonal entries are set to -inf for D1 and D2, and graph
        unreachable nodes are set to inf in D3.
    """
    # Import required libraries
    try:
        import faiss
    except ImportError:
        raise ImportError("faiss is required for distance computation. Please install it first.")
        
    try:
        from scipy.sparse import lil_matrix
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        raise ImportError("scipy and scikit-learn are required for graph distances. Please install them first.")

    # Handle input types
    if hasattr(points, 'points'):  # If points is a SetOfPoints instance
        X = points.points
    else:  # If points is a numpy array
        X = points
        
    if hasattr(voltages, 'get_voltages'):  # If voltages is a VoltageMap instance
        V = voltages.get_voltages()
    else:  # If voltages is a numpy array
        V = voltages

    if not isinstance(X, np.ndarray) or not isinstance(V, np.ndarray):
        raise ValueError("Both points and voltages must be numpy arrays or appropriate class instances")

    # Ensure inputs are float32 for faiss compatibility
    X = X.astype(np.float32)
    V = V.astype(np.float32)

    # Section 1: Compute Euclidean distances in original space
    index = faiss.IndexFlatL2(X.shape[1])  # Create L2 distance index
    index.add(X)                           # Add points to the index
    D1, _ = index.search(X, X.shape[0])
    np.fill_diagonal(D1, -np.inf)          # Set diagonal to -inf

    # Section 2: Compute distances based on voltages
    index = faiss.IndexFlatL2(V.shape[1])  # Create L2 distance index
    index.add(V)                           # Add points to the index
    D2, _ = index.search(V, V.shape[0])
    np.fill_diagonal(D2, -np.inf)          # Set diagonal to -inf

    # Section 3: Compute graph-based distances
    k = min(5, X.shape[0] - 1)  # Use 5 neighbors, ensuring k is not larger than n_points - 1
    
    # Create k-nearest neighbors graph
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
    _, indices = nbrs.kneighbors(X)
    
    # Create sparse adjacency matrix
    n = X.shape[0]
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




