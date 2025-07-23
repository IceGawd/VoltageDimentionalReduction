"""
Filter module for processing voltage maps and partitioning space.
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

if __name__ == "__main__":
    # Example usage
    random.seed(42)  # For reproducibility
    
    # Create sample data
    rng = np.random.default_rng(42)
    n_points = 20
    X = rng.normal(size=(n_points, 2))  # 20 points in 2D space
    
    # Create points with boosting weights
    # Simulate a scenario where some points are more important
    weights = rng.random(n_points)  # Random weights between 0 and 1
    weights /= weights.sum()  # Normalize to probabilities
    point_set = SetOfPoints(points=X, weights=weights)
    
    print("\nTesting weight-based filtering:")
    print(f"Original dataset size: {len(point_set)}")
    print(f"Sum of weights: {weights.sum():.2f}")  # Should be 1.0
    
    # Test weight-based filtering with different sample sizes
    for sample_size in [5, 10, 20]:
        filtered_points, mask = filter_by_weights(
            point_set,
            sample_size=sample_size,
            random_state=42
        )
        print(f"\nSampling {sample_size} points:")
        print(f"Selected {len(filtered_points)} points")
        print(f"Unique points selected: {mask.sum()}")
        
    # Example with voltage-based filtering
    print("\nTesting voltage-based filtering:")
    # Create dummy voltage map for testing
    dummy_voltages = rng.random((3, n_points))  # 3 landmarks, 20 points
    voltage_map = VoltageMap()
    for i in range(3):
        voltage_map.add_solution(i, dummy_voltages[i])
    
    # Filter points
    filtered_points_set, mask = filter_by_voltage(voltage_map, point_set, threshold=0.7)
    print(f"Kept {mask.sum()} points out of {len(point_set)}")
    
    # Generate partitions
    print("\nTesting space partitioning:")
    partitions = partition_space(voltage_map, threshold=0.7)
    for i, part in enumerate(partitions):
        print(f"Partition {i} contains {part.sum()} points")
    
    print(f"Total partitions created: {len(partitions)}")