"""
Filter Module for Voltage-Based Data Processing and Space Partitioning

This module provides tools for filtering and partitioning data based on voltage
patterns and weights. It includes three main functions and a comprehensive test suite.

Main Functions:
    filter_by_voltage(voltage_map, point_set, threshold=0.5, min_maps=1)
        Filter points based on voltage thresholds:
        - Keeps points where sufficient voltage maps exceed threshold
        - Returns filtered points and selection mask
        
    partition_space(voltage_map, threshold=0.5)
        Create space partitions based on voltage patterns:
        - Generates boolean masks for each region
        - Each region corresponds to points above threshold
        
    filter_by_weights(point_set, sample_size=None, random_state=None)
        Perform weighted random sampling:
        - Sample points based on their weights
        - Useful for boosting algorithms
        
Command Line Usage:
    python filter.py
        Runs the test suite to verify all functionality
        No additional parameters required
        Exit code 0 if successful, 1 if tests fail

Example:
    >>> from filter import filter_by_voltage, partition_space
    >>> # Filter points above 0.7 voltage threshold
    >>> filtered_points, mask = filter_by_voltage(voltage_map, points, 0.7)
    >>> # Create space partitions
    >>> regions = partition_space(voltage_map, 0.5)
"""

import numpy as np
import random
from typing import List, Tuple, Optional
from voltagemap import VoltageMap
from setofpoints import SetOfPoints

def filter_by_voltage(voltage_map: VoltageMap, 
                     point_set: SetOfPoints,
                     threshold: float = 0.5,
                     min_maps: int = 1) -> Tuple[SetOfPoints, np.ndarray]:
    """
    Filter points based on voltage map thresholds and generate space partition.

    Args:
        voltage_map (VoltageMap): Collection of voltage maps to use for filtering
        point_set (SetOfPoints): Original set of points to filter
        threshold (float, optional): Voltage threshold for filtering. Defaults to 0.5
        min_maps (int, optional): Minimum number of voltage maps that must exceed
            threshold for a point to be included. Defaults to 1

    Returns:
        Tuple[SetOfPoints, np.ndarray]: 
            - Filtered set of points
            - Binary array indicating which points were kept (1) or filtered (0)

    Example:
        >>> filtered_points, mask = filter_by_voltage(voltage_map, points, threshold=0.7)
        >>> print(f"Kept {mask.sum()} points out of {len(points)}")
    """
    # Validate threshold
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    
    # Get voltage maps as a matrix (landmarks × points)
    voltages = voltage_map.all_solutions()
    above_threshold = np.sum(voltages >= threshold, axis=1)
    
    # Create filter mask where enough maps exceed threshold
    filter_mask = above_threshold >= min_maps
    filtered_points = point_set.points[filter_mask]
    filtered_weights = point_set.weights[filter_mask] if point_set.weights is not None else None
    
    # Create new SetOfPoints with filtered data
    filtered_point_set = SetOfPoints(points=filtered_points, weights=filtered_weights)
    
    return filtered_point_set, filter_mask

def partition_space(voltage_map: VoltageMap, 
                   threshold: float = 0.5) -> List[np.ndarray]:
    """
    Generate space partition according to voltage maps.
    
    Creates partitions where each region corresponds to points where a specific
    voltage map exceeds the threshold.

    Args:
        voltage_map (VoltageMap): Collection of voltage maps to use for partitioning
        threshold (float, optional): Voltage threshold for partitioning. Defaults to 0.5

    Returns:
        List[np.ndarray]: List of boolean masks, one for each partition region
        
    Example:
        >>> partitions = partition_space(voltage_map, threshold=0.6)
        >>> for i, part in enumerate(partitions):
        ...     print(f"Partition {i} contains {part.sum()} points")
    """
    # Validate threshold
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    # Get voltage maps as a matrix (landmarks × points)
    voltages = voltage_map.all_solutions()
    
    # Create partition masks where each voltage map exceeds threshold
    partitions = []
    for i in range(voltages.shape[0]):
        mask = voltages[i, :] >= threshold
        # Make sure the mask has the right size
        if len(mask) != voltages.shape[1]:
            mask = np.zeros(voltages.shape[1], dtype=bool)
        partitions.append(mask)
    
    return partitions

def filter_by_weights(point_set: SetOfPoints,
                     sample_size: Optional[int] = None,
                     random_state: Optional[int] = None) -> Tuple[SetOfPoints, np.ndarray]:
    """
    Filter points probabilistically according to their boosting weights.
    
    This function performs weighted random sampling where points with higher weights
    are more likely to be selected. This is particularly useful in boosting algorithms
    where misclassified points receive higher weights in subsequent iterations.
    
    Args:
        point_set (SetOfPoints): Set of points with associated weights
        sample_size (int, optional): Number of points to sample. If None, uses same size
            as input. Defaults to None.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.
    
    Returns:
        Tuple[SetOfPoints, np.ndarray]: 
            - Filtered set of points
            - Binary array indicating which points were sampled (1) or not (0)
            
    Example:
        >>> points = SetOfPoints(points=X, weights=np.array([0.1, 0.7, 0.2]))
        >>> sampled_points, mask = filter_by_weights(points, sample_size=2)
        >>> print(f"Sampled {mask.sum()} points")
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_points = len(point_set)
    if sample_size is None:
        sample_size = n_points
    
    # Get or create weights
    weights = point_set.weights
    if weights is None:
        weights = np.ones(n_points) / n_points
    else:
        # Normalize weights to probabilities
        weights = weights / weights.sum()
    
    # Perform weighted random sampling
    selected_indices = np.random.choice(
        n_points, 
        size=sample_size, 
        p=weights,
        replace=True  # Allow repeated selection
    )
    
    # Create mask of selected points
    mask = np.zeros(n_points, dtype=bool)
    mask[selected_indices] = True
    
    # Create new SetOfPoints with sampled data
    filtered_points = point_set.points[selected_indices]
    filtered_weights = weights[selected_indices] if weights is not None else None
    filtered_point_set = SetOfPoints(points=filtered_points, weights=filtered_weights)
    
    return filtered_point_set, mask

def test_filter_functions():
    """
    Run comprehensive tests for all filter functions with deterministic data.
    Verifies functionality of voltage-based filtering, space partitioning,
    and weight-based filtering.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    try:
        print("\nTesting filter functions...")
        random.seed(42)  # For reproducibility
        rng = np.random.default_rng(42)
        
        # Test 1: Weight-based filtering
        print("\nTesting weight-based filtering:")
        n_points = 20
        X = rng.normal(size=(n_points, 2))
        weights = np.array([0.1, 0.3, 0.4, 0.2] * 5)  # Deterministic weights
        point_set = SetOfPoints(points=X, weights=weights)
        
        # Test different sample sizes
        sample_sizes = [5, 10, 20]
        
        for sample_size in sample_sizes:
            filtered_points, mask = filter_by_weights(
                point_set,
                sample_size=sample_size,
                random_state=42
            )
            # Check that we got the right number of sampled points
            assert len(filtered_points) == sample_size, \
                f"Expected {sample_size} sampled points, got {len(filtered_points)}"
            
            # Check that mask is boolean and has correct length
            assert mask.dtype == bool, "Mask should be boolean type"
            assert len(mask) == n_points, f"Mask should have length {n_points}"
            
            # Check that mask sum is less than or equal to sample_size (due to possible duplicates)
            assert mask.sum() <= sample_size, \
                f"Mask sum ({mask.sum()}) should not exceed sample size ({sample_size})"
            
            # Check weight normalization
            assert np.isclose(filtered_points.weights.sum(), 1.0), \
                "Weights should sum to 1.0"
            
            # Verify that selected points come from higher weight regions more often
            if sample_size >= 10:  # Only check for larger samples
                high_weight_indices = np.where(weights >= 0.3)[0]
                selected_high_weight = np.sum(mask[high_weight_indices])
                assert selected_high_weight > 0, \
                    "Should select some points from high weight regions"
        print(" Weight-based filtering tests passed")
        
        # Test 2: Voltage-based filtering
        print("\nTesting voltage-based filtering:")
        # Create deterministic voltage maps
        voltage_values = np.array([
            [0.8, 0.3, 0.6, 0.9] * 5,  # Some above 0.7 threshold
            [0.2, 0.9, 0.1, 0.7] * 5,  # Different pattern above 0.7
            [0.4, 0.5, 0.8, 0.6] * 5   # Another pattern
        ])
        
        voltage_map = VoltageMap()
        for i in range(3):
            voltage_map.add_solution(i, voltage_values[i])
            
        # Test with different thresholds
        thresholds = [0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            filtered_points, mask = filter_by_voltage(voltage_map, point_set, threshold=threshold)
            
            # Compute expected result manually
            voltages_above = voltage_values >= threshold
            expected_mask = np.sum(voltages_above, axis=0) >= 1  # At least one map above threshold
            expected_count = np.sum(expected_mask)
            
            assert mask.sum() == expected_count, \
                f"Expected {expected_count} points for threshold {threshold}, got {mask.sum()}"
            
            # Verify that all points in the mask actually exceed the threshold in at least one map
            for i in range(len(mask)):
                if mask[i]:
                    assert any(voltage_values[:, i] >= threshold), \
                        f"Point {i} shouldn't be in mask for threshold {threshold}"
        print(" Voltage-based filtering tests passed")
        
        # Test 3: Space partitioning
        print("\nTesting space partitioning:")
        
        # Create a small test case with known values
        test_voltages = np.array([
            [0.8, 0.3, 0.6],  # First voltage map
            [0.2, 0.9, 0.1],  # Second voltage map
            [0.4, 0.5, 0.8]   # Third voltage map
        ])
        
        test_voltage_map = VoltageMap()
        for i in range(3):
            test_voltage_map.add_solution(i, test_voltages[i])
        
        # Test partitioning with threshold
        threshold = 0.7
        partitions = partition_space(test_voltage_map, threshold=threshold)
        
        # Verify partitions
        assert len(partitions) == 3, f"Expected 3 partitions, got {len(partitions)}"
        
        # Expected results for each partition (where voltage >= 0.7)
        expected_partitions = [
            [True, False, False],  # First map: only first point >= 0.7
            [False, True, False],  # Second map: only second point >= 0.7
            [False, False, True]   # Third map: only third point >= 0.7
        ]
        
        for i, part in enumerate(partitions):
            assert len(part) == 3, f"Partition {i} should have length 3, got {len(part)}"
            assert part.dtype == bool, f"Partition {i} should be boolean mask, got {part.dtype}"
            assert np.array_equal(part, expected_partitions[i]), \
                f"Partition {i} has incorrect mask values"
        
        print(" Space partitioning tests passed")
        
        # Test 4: Edge cases
        print("\nTesting edge cases:")
        
        # Test empty weights
        no_weights_set = SetOfPoints(points=X)
        filtered_no_weights, mask = filter_by_weights(no_weights_set, sample_size=5, random_state=42)
        assert len(filtered_no_weights) == 5, "Should handle points without weights"
        
        # Test threshold validation
        try:
            filter_by_voltage(voltage_map, point_set, threshold=1.5)
            assert False, "Should raise error for threshold > 1"
        except ValueError:
            pass
        
        try:
            filter_by_voltage(voltage_map, point_set, threshold=-0.5)
            assert False, "Should raise error for threshold < 0"
        except ValueError:
            pass
            
        print(" Edge cases tests passed")
        
        print("\nAll filter function tests passed successfully!")
        return True
        
    except AssertionError as e:
        print(f"Test failed: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_filter_functions()
    if not success:
        print("\nSome tests failed!")
        exit(1)
    print("\nAll tests completed successfully!")