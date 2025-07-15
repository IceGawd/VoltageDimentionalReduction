"""
Filter module for processing voltage maps and partitioning space.
"""

import numpy as np
import random
from typing import List, Tuple, Optional
from voltagemap import VoltageMap
from setofpoints import SetOfPoints
from visualization import Visualization

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

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_blobs
    random.seed(42)  # For reproducibility
    
    # Create synthetic test data
    X, y = make_blobs(n_samples=20, centers=3, random_state=42)
    point_set = SetOfPoints(points=X)
    
    # Dummy voltage map for testing
    # In real usage, this would come from solving the voltage equations
    dummy_voltages = np.random.rand(3, 20)  # 3 landmarks, 20 points
    voltage_map = VoltageMap()
    for i in range(3):
        voltage_map.add_solution(i, dummy_voltages[i])
    
    # Filter points
    filtered_points_set, mask = filter_by_voltage(voltage_map, point_set, threshold=0.7)
    print(f"Kept {mask.sum()} points out of {len(point_set)}")
    
    # Generate partitions
    partitions = partition_space(voltage_map, threshold=0.7)
    for i, part in enumerate(partitions):
        print(f"Partition {i} contains {part.sum()} points")
    
    print(f"Total partitions created: {len(partitions)}")
    