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
    # Check if required packages are available
    if not HAVE_FAISS:
        raise ImportError("faiss is required for distance computation. Please install it first.")
    if not HAVE_SCIPY or not HAVE_SKLEARN:
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
    n_points = X.shape[0]
    
    # Compute pairwise squared Euclidean distances using broadcasting
    X_expanded = X[:, np.newaxis, :]  # Shape: (n_points, 1, n_dims)
    X_T_expanded = X[np.newaxis, :, :]  # Shape: (1, n_points, n_dims)
    diff = X_expanded - X_T_expanded  # Shape: (n_points, n_points, n_dims)
    D1 = np.sum(diff * diff, axis=2).astype(np.float32)  # Sum over last axis
    
    np.fill_diagonal(D1, -np.inf)  # Set diagonal to -inf

    # Section 2: Compute distances based on voltages
    # Compute pairwise squared Euclidean distances using broadcasting
    V_expanded = V[:, np.newaxis, :]  # Shape: (n_points, 1, n_voltage_dims)
    V_T_expanded = V[np.newaxis, :, :]  # Shape: (1, n_points, n_voltage_dims)
    diff = V_expanded - V_T_expanded  # Shape: (n_points, n_points, n_voltage_dims)
    D2 = np.sum(diff * diff, axis=2).astype(np.float32)  # Sum over last axis
    
    np.fill_diagonal(D2, -np.inf)  # Set diagonal to -inf

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


# ------------------- Test Function ---------------------
def test_compute_distances():
    """
    Test function for compute_distances with deterministic data.
    
    This function creates synthetic test data with known properties and verifies
    that the distance computation functions work correctly. It tests:
    1. Basic functionality with simple test cases
    2. Edge cases (small datasets, identical points)
    3. Consistency of distance matrices (symmetry, diagonal properties)
    4. Correctness of different distance metrics
    
    The test uses deterministic data to ensure reproducible results across runs.
    
    Returns:
        bool: True if all tests pass, False otherwise.
        
    Example Output:
        Testing compute_distances...
         Basic functionality test passed
         Distance matrix properties test passed
         Edge cases test passed
         Voltage distance test passed
         Graph distance test passed
        All tests passed successfully!
    """
    print("Testing compute_distances...")
    
    try:
        # Test 1: Basic functionality with deterministic data
        print("Running basic functionality test...")
        np.random.seed(42)  # Ensure reproducible results
        
        # Create test points in 2D space with known distances
        test_points = np.array([
            [0.0, 0.0],   # Origin
            [1.0, 0.0],   # Unit distance on x-axis
            [0.0, 1.0],   # Unit distance on y-axis
            [1.0, 1.0],   # Corner point
            [0.5, 0.5]    # Center point
        ], dtype=np.float32)
        
        # Create test voltages with known structure
        test_voltages = np.array([
            [1.0, 0.0, 0.0],  # High voltage in first dimension
            [0.0, 1.0, 0.0],  # High voltage in second dimension
            [0.0, 0.0, 1.0],  # High voltage in third dimension
            [0.5, 0.5, 0.0],  # Mixed voltages
            [0.3, 0.3, 0.3]   # Balanced voltages
        ], dtype=np.float32)
        
        # Compute distances
        D1, D2, D3 = compute_distances(test_points, test_voltages)
        
        # Verify shapes
        expected_shape = (5, 5)
        assert D1.shape == expected_shape, f"D1 shape mismatch: {D1.shape} != {expected_shape}"
        assert D2.shape == expected_shape, f"D2 shape mismatch: {D2.shape} != {expected_shape}"
        assert D3.shape == expected_shape, f"D3 shape mismatch: {D3.shape} != {expected_shape}"
        print(" Basic functionality test passed")
        
        # Test 2: Distance matrix properties
        print("Running distance matrix properties test...")
        
        # Check diagonal elements (should be -inf for D1 and D2)
        for i in range(5):
            assert D1[i, i] == -np.inf, f"D1 diagonal not -inf at [{i}, {i}]: {D1[i, i]}"
            assert D2[i, i] == -np.inf, f"D2 diagonal not -inf at [{i}, {i}]: {D2[i, i]}"
            assert D3[i, i] == 0.0, f"D3 diagonal not 0 at [{i}, {i}]: {D3[i, i]}"
        
        # Check symmetry (distances should be symmetric)
        # Use a more appropriate tolerance for floating point comparison
        tolerance = 1e-5  # Increased tolerance for floating point precision
        for i in range(5):
            for j in range(5):
                if i != j:
                    # Skip comparisons involving -inf (diagonal elements that were reset)
                    if not (np.isinf(D1[i, j]) or np.isinf(D1[j, i])):
                        assert abs(D1[i, j] - D1[j, i]) < tolerance, f"D1 not symmetric at [{i}, {j}]: {D1[i, j]} != {D1[j, i]}"
                    if not (np.isinf(D2[i, j]) or np.isinf(D2[j, i])):
                        assert abs(D2[i, j] - D2[j, i]) < tolerance, f"D2 not symmetric at [{i}, {j}]: {D2[i, j]} != {D2[j, i]}"
                    if not (np.isinf(D3[i, j]) or np.isinf(D3[j, i])):
                        assert abs(D3[i, j] - D3[j, i]) < tolerance, f"D3 not symmetric at [{i}, {j}]: {D3[i, j]} != {D3[j, i]}"
        
        # Check some known distances in original space
        # Distance from (0,0) to (1,0) should be 1.0
        expected_dist = 1.0
        actual_dist = np.sqrt(D1[0, 1])  # FAISS returns squared distances
        assert abs(actual_dist - expected_dist) < 1e-5, f"Distance (0,0) to (1,0): {actual_dist} != {expected_dist}"
        
        # Distance from (0,0) to (1,1) should be sqrt(2)
        expected_dist = np.sqrt(2.0)
        actual_dist = np.sqrt(D1[0, 3])
        assert abs(actual_dist - expected_dist) < 1e-5, f"Distance (0,0) to (1,1): {actual_dist} != {expected_dist}"
        
        print(" Distance matrix properties test passed")
        
        # Test 3: Edge cases
        print("Running edge cases test...")
        
        # Test with minimum number of points (2 points)
        small_points = test_points[:2]
        small_voltages = test_voltages[:2]
        D1_small, D2_small, D3_small = compute_distances(small_points, small_voltages)
        
        assert D1_small.shape == (2, 2), f"Small D1 shape: {D1_small.shape}"
        assert D2_small.shape == (2, 2), f"Small D2 shape: {D2_small.shape}"
        assert D3_small.shape == (2, 2), f"Small D3 shape: {D3_small.shape}"
        
        print(" Edge cases test passed")
        
        # Test 4: Voltage distance specifics
        print("Running voltage distance test...")
        
        # Points with identical voltages should have distance 0
        identical_voltages = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ], dtype=np.float32)
        
        identical_points = np.array([
            [0.0, 0.0],
            [1.0, 1.0]
        ], dtype=np.float32)
        
        _, D2_identical, _ = compute_distances(identical_points, identical_voltages)
        voltage_dist = D2_identical[0, 1]
        assert voltage_dist == 0.0, f"Identical voltages should have distance 0: {voltage_dist}"
        
        print(" Voltage distance test passed")
        
        # Test 5: Graph distance properties
        print("Running graph distance test...")
        
        # For a small connected graph, all distances should be finite
        finite_distances = D3[D3 != 0]  # Exclude diagonal
        assert np.all(np.isfinite(finite_distances)), "Graph should be connected with finite distances"
        
        # Direct neighbors should have distance 1
        neighbor_distances = D3[D3 == 1.0]
        assert len(neighbor_distances) > 0, "Should have some direct neighbor connections"
        
        print(" Graph distance test passed")
        
        print("All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ------------------- Main ---------------------
def main():
    """
    Main function for testing the distances module.
    
    This function runs the comprehensive test suite for the compute_distances function.
    It can be called directly or used as part of a larger test framework.
    
    The test covers various scenarios and edge cases to ensure the reliability
    of the distance computation functionality.
    """
    success = test_compute_distances()
    if success:
        print("\n All distance computation tests passed!")
    else:
        print("\n Some tests failed. Please check the implementation.")
        exit(1)


if __name__ == "__main__":
    main()




