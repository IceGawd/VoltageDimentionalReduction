"""
Main script that applies voltage-based filtering before running the main analysis.

This script extends main_yoav.py by adding a preprocessing step that filters
the input data based on voltage map thresholds before proceeding with the analysis.
"""

import numpy as np
from .filter import filter_by_voltage, partition_space
from .voltagemap import VoltageMap
from .setofpoints import SetOfPoints
from .landmark import Landmark
from .solver import Solver
from .problem import Problem

def run_analysis_on_filtered_data(
    point_set: SetOfPoints,
    voltage_threshold: float = 0.5,
    min_maps: int = 1,
    **kwargs
) -> Tuple[SetOfPoints, VoltageMap, List[np.ndarray]]:
    """
    Run analysis on filtered data.

    Args:
        point_set (SetOfPoints): Input points to analyze
        voltage_threshold (float, optional): Threshold for filtering. Defaults to 0.5
        min_maps (int, optional): Minimum maps above threshold. Defaults to 1
        **kwargs: Additional arguments passed to the solver

    Returns:
        Tuple[SetOfPoints, VoltageMap, List[np.ndarray]]:
            - Filtered point set
            - Voltage map for filtered points
            - List of partition masks
    """
    # Create initial problem and compute voltage maps
    problem = Problem(point_set)
    solver = Solver(problem)
    
    # Create landmarks (this should be customized based on your needs)
    landmarks = [Landmark(i) for i in range(min_maps)]
    
    # Compute initial voltage maps
    voltage_map = VoltageMap()
    for lm in landmarks:
        voltages = solver.compute_voltages(lm)
        voltage_map.add_solution(lm, voltages)
    
    # Filter points based on voltage maps
    filtered_points, filter_mask = filter_by_voltage(
        voltage_map,
        point_set,
        threshold=voltage_threshold,
        min_maps=min_maps
    )
    
    # Generate space partitions
    partitions = partition_space(voltage_map, threshold=voltage_threshold)
    
    # Recompute problem and voltage maps for filtered points
    filtered_problem = Problem(filtered_points)
    filtered_solver = Solver(filtered_problem)
    
    filtered_voltage_map = VoltageMap()
    for lm in landmarks:
        voltages = filtered_solver.compute_voltages(lm)
        filtered_voltage_map.add_solution(lm, voltages)
    
    return filtered_points, filtered_voltage_map, partitions

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_blobs
    
    # Create synthetic test data
    X, y = make_blobs(n_samples=1000, centers=5, random_state=42)
    point_set = SetOfPoints(points=X)
    
    # Run analysis with filtering
    filtered_points, voltage_map, partitions = run_analysis_on_filtered_data(
        point_set,
        voltage_threshold=0.7,
        min_maps=2
    )
    
    # Print results
    print(f"Original points: {len(point_set)}")
    print(f"Filtered points: {len(filtered_points)}")
    print(f"Number of partitions: {len(partitions)}")
    for i, part in enumerate(partitions):
        print(f"Partition {i} size: {part.sum()}")
    
    # Additional analysis based on main_yoav.py
