"""
Landmark Selection Algorithm for Voltage-based Dimensionality Reduction

This module implements a greedy algorithm for selecting landmarks based on voltage
distributions. It aims to find a diverse set of landmarks that provide good
coverage of the data space by maximizing distances between voltage patterns.

The selection process uses both voltage distribution differences and landmark
quality metrics to choose landmarks that are:
1. Well-separated from existing selections
2. Have high quality scores (norms)
3. Provide complementary information about the data structure

Example:
    >>> from select_landmarks import select_landmarks
    >>> # Assuming we have computed voltage patterns
    >>> selected_map = select_landmarks(all_voltage_patterns)
    >>> print(f"Selected {len(selected_map)} landmarks")
"""

import voltagemap   
import numpy as np
from typing import Optional

def select_landmarks(all_voltages: voltagemap.VoltageMap) -> voltagemap.VoltageMap:
    """
    Select a subset of landmarks using a greedy diversity-based approach.

    This function implements an iterative selection process that:
    1. Starts with the first landmark as initialization
    2. Iteratively selects new landmarks that are:
       - Maximally distant from already selected landmarks
       - Have high quality scores (norms)
    3. Continues until no more suitable landmarks are found

    The selection criteria use two thresholds:
    - Distance threshold (1.3): Minimum distance to existing landmarks
    - Norm threshold (1.3): Minimum quality score for a landmark

    Args:
        all_voltages (VoltageMap): Complete set of candidate landmarks with
            their voltage patterns. Each entry contains:
            - landmark: The landmark point
            - voltages: The computed voltage distribution
            - norm: Quality score for the landmark

    Returns:
        VoltageMap: Selected subset of landmarks that:
            - Are well-distributed (high mutual distances)
            - Have high quality scores
            - Provide good coverage of the data space

    Note:
        This is an experimental implementation and may be improved. The
        current thresholds (1.3) are empirically chosen and might need
        adjustment for different datasets.

    Example:
        >>> # Assuming we have computed voltage patterns
        >>> all_patterns = compute_all_voltage_patterns(data)
        >>> selected = select_landmarks(all_patterns)
        >>> print(f"Selected {len(selected)} diverse landmarks")
    """
    # Initialize the map
    voltage_map=voltagemap.VoltageMap()
    lm, voltages, _ = all_voltages.entries[0]  # get the first landmark and its voltages
    voltage_map.add_solution(lm, voltages=voltages)
    max_voltage=np.zeros(len(all_voltages))  # to keep track of the maximum voltage for each landmark

    # repeatedly iteration all_voltages.entries and add the landmark with the largest distance to the selected landmarks to the voltage map	
    for iteration in range(100):
        # Find the landmark in all_voltages.entries that is farthest from the current voltage_map entries
        max_min_dist = 1.3
        best_idx = None
        best_norm = 1.3
        for idx, (lm, voltages, norm) in enumerate(all_voltages.entries):
            # Skip if already in voltage_map
            if any(np.array_equal(lm.index, vmap_lm.index) for vmap_lm, _, _ in voltage_map.entries):
                continue	
            # Compute minimum distance to any entry in voltage_map
            min_dist = np.min([np.linalg.norm(voltages - vm[1]) for vm in voltage_map.entries])
            if min_dist > max_min_dist and norm> best_norm:
                max_min_dist = min_dist
                best_idx = idx
                best_norm = norm
        print(f"Iteration {iteration}: Best landmark index {best_idx} norm={best_norm:.4f} with min distance {max_min_dist:.4f}")
        if best_idx is not None:
            lm, voltages, norm = all_voltages.entries[best_idx]
            voltage_map.add_solution(lm, voltages=voltages)
        else:
            break
    return voltage_map
