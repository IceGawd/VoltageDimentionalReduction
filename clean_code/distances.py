"""
Distance computation module for point sets and voltage data.

This module provides functionality to compute various distance metrics between points
in a dataset, including Euclidean distances, voltage-based distances, and graph-based
distances using k-nearest neighbors connectivity.

The module uses FAISS for efficient similarity search and distance computation,
particularly useful for large-scale datasets.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.sparse import csgraph, lil_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
import faiss
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from clean_code.Utilities import config
from clean_code.Utilities.set_params import set_params
from clean_code.setofpoints import SetOfPoints

def create_faiss_index(dim: int, use_gpu: bool = False) -> faiss.Index:
    """
    Create an optimized FAISS index based on data dimension and available hardware.
    
    Args:
        dim (int): Dimensionality of the vectors
        use_gpu (bool): Whether to use GPU if available
        
    Returns:
        faiss.Index: Optimized FAISS index
    """
    if use_gpu and hasattr(faiss, 'StandardGpuResources'):
        res = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.device = 0  # Use first GPU
        return faiss.GpuIndexFlatL2(res, dim, config)
    else:
        if dim <= 4:
            return faiss.IndexFlatL2(dim)  # Exact search for low dimensions
        else:
            index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors per layer
            index.hnsw.efConstruction = 40  # Higher accuracy construction
            index.hnsw.efSearch = 16  # Compromise between speed and accuracy
            return index

def compute_distances(point_set: SetOfPoints, voltages: np.ndarray, 
                     use_gpu: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute three types of distances between points using FAISS for efficiency.
    
    This function computes:
    1. Euclidean distances between points in the point set
    2. Distances between points based on their voltage values
    3. Graph-based distances using k-nearest neighbors connectivity
       (measured as the number of hops in the graph)
       
    The implementation is optimized for:
    - GPU acceleration when available
    - Efficient memory usage with sparse matrices
    - Automatic index selection based on data dimensionality
    - Parallel computation of distances where possible
    
    Args:
        point_set (SetOfPoints): Set of points in the original space.
        voltages (np.ndarray): Matrix of voltage values for each point.
            Shape should be (n_points, n_voltages).
            
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Three distance matrices:
            - D1: Euclidean distances between original points
            - D2: Distances based on voltage values
            - D3: Graph-based distances (number of hops)
            
    Raises:
        ValueError: If point_set and voltages have incompatible dimensions.
        RuntimeError: If FAISS index building fails.
        
    Note:
        - All distance matrices have -inf on the diagonal
        - The k-connectivity graph is undirected
        - Unreachable nodes in the graph have distance = inf
        
    Example:
        >>> points = SetOfPoints(np.random.rand(100, 2))
        >>> voltages = np.random.rand(100, 10)
        >>> D1, D2, D3 = compute_distances(points, voltages)
        >>> print(f"Distance matrices shapes: {D1.shape}, {D2.shape}, {D3.shape}")
    """
    # Validate input dimensions
    if len(point_set.points) != len(voltages):
        raise ValueError("Number of points must match number of voltage values")
    
    # Convert inputs to float32 once
    points = point_set.points.astype(np.float32)
    volts = voltages.astype(np.float32)
    n = len(points)
    
    D1 = np.empty((n, n), dtype=np.float32)
    D2 = np.empty((n, n), dtype=np.float32)
    
    # 1. Euclidean distance based on original points
    index1 = create_faiss_index(points.shape[1], use_gpu)
    index1.add(points)
    D1, _ = index1.search(points, n)
    D1 = np.minimum(D1, D1.T)
    np.fill_diagonal(D1, -np.inf)
    
    # 2. Distance based on voltages
    del index1
    index2 = create_faiss_index(volts.shape[1], use_gpu)
    index2.add(volts)
    D2, _ = index2.search(volts, n)
    D2 = np.minimum(D2, D2.T)
    np.fill_diagonal(D2, -np.inf)
    del index2  

    # 3. Distance based on k-connectivity graph
    if n == 1:
        # Special case: single point
        D3 = np.array([[-np.inf]])
    else:
        # Adjust k to be at most n-1 (all other points)
        k = min(config.params['k'], n - 1)
        
        # Reuse FAISS index for k-nearest neighbors if dimension is suitable
        if volts.shape[1] <= 32:  # For low dimensions, FAISS is very efficient
            index3 = create_faiss_index(volts.shape[1], use_gpu)
            index3.add(volts)
            _, indices = index3.search(volts, k + 1)  # +1 because first match is self
            indices = indices[:, 1:]  # Remove self-matches
            del index3
        else:
            # For high dimensions, sklearn might be more memory efficient
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(volts)
            _, indices = nbrs.kneighbors()
        
        # Create sparse adjacency matrix directly in CSR format for efficiency
        indptr = np.arange(0, n * k + 1, k)
        indices_flat = indices.ravel()
        data = np.ones_like(indices_flat, dtype=np.float32)
        
        # Create symmetric adjacency matrix in one shot
        adjacency = csr_matrix(
            (data, indices_flat, indptr),
            shape=(n, n)
        )
        adjacency = adjacency.maximum(adjacency.T)  # Symmetrize efficiently
        
        # Compute shortest paths optimized for sparse unweighted graph
        D3 = csgraph.shortest_path(
            adjacency, method='D', directed=False, 
            unweighted=True, return_predecessors=False
        )

        mask = np.isinf(D3)
        D3[mask] = np.inf
        np.fill_diagonal(D3, -np.inf)

    return D1, D2, D3

def test_distances():
    """
    Run comprehensive tests for distance computation functionality.
    
    This function tests the distance computation with various input cases including:
    - Small deterministic datasets
    - Edge cases (single point, identical points)
    - Different dimensionalities
    - Graph connectivity cases
    
    Returns:
        bool: True if all tests pass, False otherwise.
    """
    try:
        print("\nTesting distance computations...")
        np.random.seed(42)  # For reproducibility
        
        # Test 1: Simple deterministic case
        print("\nTest 1: Simple deterministic case")
        points = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1]
        ], dtype=np.float32)
        voltages = np.array([
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
            [0.4, 0.5]
        ], dtype=np.float32)
        
        point_set = SetOfPoints(points=points)
        D1, D2, D3 = compute_distances(point_set, voltages)
        
        # Verify matrix shapes
        assert D1.shape == (4, 4), f"Expected shape (4,4), got {D1.shape}"
        assert D2.shape == (4, 4), f"Expected shape (4,4), got {D2.shape}"
        assert D3.shape == (4, 4), f"Expected shape (4,4), got {D3.shape}"
        
        # Verify diagonal values
        assert np.all(np.diag(D1) == -np.inf), "D1 diagonal should be -inf"
        assert np.all(np.diag(D2) == -np.inf), "D2 diagonal should be -inf"
        
        # Verify symmetry
        assert np.allclose(D1, D1.T), "D1 should be symmetric"
        assert np.allclose(D2, D2.T), "D2 should be symmetric"
        assert np.allclose(D3, D3.T), "D3 should be symmetric"
        
        print(" Simple case tests passed")
        
        # Test 2: Edge cases
        print("\nTest 2: Edge cases")
        # Single point
        single_point = SetOfPoints(points=np.array([[1.0, 1.0]], dtype=np.float32))
        single_voltage = np.array([[0.5, 0.5]], dtype=np.float32)
        D1, D2, D3 = compute_distances(single_point, single_voltage)
        assert D1.shape == (1, 1), "Single point should give 1x1 matrix"
        assert D1[0,0] == -np.inf, "Single point diagonal should be -inf"
        
        # Identical points
        identical_points = SetOfPoints(points=np.tile(np.array([1.0, 1.0]), (3, 1)))
        identical_voltages = np.tile(np.array([0.5, 0.5]), (3, 1))
        D1, D2, D3 = compute_distances(identical_points, identical_voltages)
        assert np.all(D1[~np.isinf(D1)] == 0), "Identical points should have zero distance in D1"
        assert np.all(D2[~np.isinf(D2)] == 0), "Identical points should have zero distance in D2"
        assert D3.shape == (3, 3), "D3 should be 3x3 for identical points"
        assert np.all(np.diag(D3) == -np.inf), "D3 diagonal should be -inf for identical points"
        
        print(" Edge cases passed")
        
        # Test 3: Different dimensionalities
        print("\nTest 3: Testing different dimensionalities")
        # High-dimensional points
        high_dim_points = SetOfPoints(points=np.random.rand(10, 50).astype(np.float32))
        high_dim_voltages = np.random.rand(10, 5).astype(np.float32)
        D1, D2, D3 = compute_distances(high_dim_points, high_dim_voltages)
        assert D1.shape == (10, 10), "Wrong shape for high-dimensional case"
        
        print(" Dimensionality tests passed")
        
        # Test 4: Graph connectivity
        print("\nTest 4: Testing graph connectivity")
        # Create a disconnected graph case
        points_disconnected = np.vstack([
            np.random.rand(5, 2),
            np.random.rand(5, 2) + np.array([10, 10])  # Separated cluster
        ]).astype(np.float32)
        voltages_disconnected = np.random.rand(10, 3).astype(np.float32)
        
        config.params['k'] = 2  # Set small k to ensure disconnected components
        D1, D2, D3 = compute_distances(SetOfPoints(points=points_disconnected), 
                                     voltages_disconnected)
        
        # Verify that some points are unreachable (inf distance)
        assert np.any(np.isinf(D3)), "Disconnected graph should have inf distances"
        
        print(" Graph connectivity tests passed")
        
        # Test 5: Input validation
        print("\nTest 5: Testing input validation")
        try:
            # Mismatched dimensions
            bad_voltages = np.random.rand(5, 2)
            _ = compute_distances(SetOfPoints(points=np.random.rand(4, 2)), bad_voltages)
            assert False, "Should raise ValueError for mismatched dimensions"
        except ValueError:
            print(" Input validation tests passed")
        
        print("\nAll distance computation tests passed successfully!")
        return True
        
    except AssertionError as e:
        print(f"Test failed: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    # Initialize configuration with defaults if not already set
    if not hasattr(config, 'params'):
        config.params = {}
    
    # Set required parameters for testing
    config.params.setdefault('k', 3)  # Default k for nearest neighbors
    config.params.setdefault('test', True)
    config.params.setdefault('normalize_vectors', False)  # Default to no normalization
    
    # Run tests
    success = test_distances()
    if not success:
        print("\nSome tests failed!")
        exit(1)
    print("\nAll tests completed successfully!")




