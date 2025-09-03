"""
Mutual Information-based Landmark Selection

This module implements an information-theoretic approach to landmark selection
for dimensionality reduction. It uses mutual information (MI) between voltage
patterns and centroid identities to select landmarks that maximize the
information content about the data structure.

The algorithm employs a greedy search strategy to:
1. Evaluate individual landmarks based on their MI scores
2. Incrementally build a set of landmarks that jointly maximize MI
3. Consider interactions between landmarks through cumulative MI

Configuration (via config.params):
    NoOfLandmarks (int): Number of landmarks to select
    DepthOfLandmarkSearch (int): How many candidates to consider at each step

Example:
    >>> from select_landmarks_MI import select_landmarks
    >>> # Assuming we have voltage patterns
    >>> selected = select_landmarks(all_voltages)
    >>> print(f"Selected {len(selected)} informative landmarks")
"""

import voltagemap   
import numpy as np
from Utilities import config
from copy import copy
from Mutual_Information import mutual_information
from Utilities.set_params import set_params
from Utilities.timer import Timer
from typing import Optional

def select_landmarks(all_voltages: voltagemap.VoltageMap) -> voltagemap.VoltageMap:
    """
    Select landmarks using mutual information maximization.

    This function implements a greedy algorithm that selects landmarks based on
    their mutual information (MI) with centroid identities. The selection process:
    1. Computes individual MI scores for each landmark
    2. Sorts landmarks by their individual informativeness
    3. Greedily adds landmarks that maximize cumulative MI

    Algorithm Steps:
        1. Initialize: Compute MI_1 (individual MI scores) for all landmarks
        2. Sort by MI_1 to prioritize individually informative landmarks
        3. Iteratively select N landmarks:
           - For each step, evaluate MI of candidate combinations
           - Choose landmark that maximizes cumulative MI
           - Update rankings and continue

    Args:
        all_voltages (VoltageMap): Complete set of voltage patterns where each
            entry contains:
            - landmark: The landmark point
            - voltages: Computed voltage distribution
            - MI_1: Will store individual MI score
            - MI_cumul: Will store cumulative MI score

    Returns:
        VoltageMap: Selected subset of landmarks that:
            - Maximizes mutual information with centroid identities
            - Contains exactly N landmarks (from config)
            - Is ordered by cumulative MI contribution

    Raises:
        ValueError: If requested number of landmarks exceeds available landmarks

    Note:
        Uses parameters from config:
        - NoOfLandmarks (N): Number of landmarks to select
        - DepthOfLandmarkSearch (D): Number of candidates to evaluate at each step
    """
    timer = Timer()
    timer.mark("Starting landmark selection based on mutual information")
    # Initialize the map
    voltage_map=copy(all_voltages)
    voltage_map.set_advantages(-np.inf,quantity="MI_cumul")  # Set initial advantages to negative infinity
    L=len(voltage_map.entries)
    N=config.params['NoOfLandmarks']
    if N>L:
        raise ValueError(f"Number of landmarks {N} exceeds the number of centroids {L}.") 

    # initialize the quantity "MI_1" to be the mutual information for each landmark by itself
    for i in range(L):
        voltage_matrix = np.stack([voltage_map.entries[i]['voltages']],axis=1)
        # Compute mutual information (MI) between voltage_matrix and the identity of the centroid
        mi = mutual_information(voltage_matrix)
        voltage_map.entries[i]['MI_1'] = mi  # Set the advantage to the mutual
        if(i%10==0):
            print(f"Landmark index={i} MI_1={mi:.4f}",end='\r')
    timer.mark(f"calculated MI for each singleton")

    voltage_map.sort_by_advantage(quantity="MI_1", reverse=True)  # Sort the voltage map by MI_1

    D=config.params['DepthOfLandmarkSearch']
    # repeatedly iteration all_voltages.entries and add the landmark with the largest distance to the selected landmarks to the voltage map	
    for i in range(N):
        # find the landmark whose addition to the voltage map will increase the mutual information by the max amount
        max_mi = -np.inf
        for j in range(i,D):
            voltage_matrix = np.stack([voltage_map.entries[k]['voltages'] for k in (list(range(i))+[j]) ],axis=1)  # Stack the voltages of the current landmarks and the new candidate landmark
            # Compute mutual information (MI) between voltage_matrix and the identity of the centroid
            mi = mutual_information(voltage_matrix)
            #print(f"Landmarks 0:{i}+{j} MI={mi:.4f}")
            if mi > max_mi:
                max_mi = mi
                best_landmark = j

        # move the best landmark to the voltage map
        print(f"Adding landmark index={best_landmark} lm_index={voltage_map.entries[best_landmark]['landmark'].index} with MI={max_mi:.4f}")
        voltage_map.entries[best_landmark]['MI_cumul'] = max_mi  # Update the advantage of the best landmark
        # Sort the voltage map by advantage
        voltage_map.sort_by_advantage(quantity="MI_cumul", reverse=True)
    timer.mark(f"Selected {N} landmarks based on mutual information")
    # remove landmarks beyond N
    voltage_map.entries = voltage_map.entries[:N]   
    return voltage_map

if __name__ == "__main__": 
    """
    Main execution script for landmark selection.

    This script:
    1. Loads parameters from command line or defaults
    2. Reads voltage patterns from a pickle file
    3. Performs MI-based landmark selection
    4. Saves selected landmarks back to the same file

    The pickle file should contain a dictionary with:
    - 'all_voltages': VoltageMap with all computed patterns
    - 'voltage_map': Will be added/updated with selected landmarks

    Configuration:
        Use command line arguments to set:
        - save_data: Path to pickle file with voltage patterns
        - NoOfLandmarks: Number of landmarks to select
        - DepthOfLandmarkSearch: Search depth for selection
    """
    set_params()  # Set parameters from command line or default values

    import pickle
    save_data = config.params['save_data']
    
    with open(save_data, 'rb') as f:
        Data=pickle.load(f)
    print(f"Data loaded from {save_data}")
    
    all_voltages = Data['all_voltages']
    centroids = Data['centroids']
    voltage_map = select_landmarks(all_voltages)

    Data['voltage_map'] = voltage_map

    from filter import count_neighborhoods
    neighborhood = count_neighborhoods(input_path=config.params['file_path'], voltage_map=voltage_map, centroids=centroids)
    Data['neighborhood'] = neighborhood

    #pretty print the neighborhood matrix and the sum for each row.
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print("Neighborhood matrix (rows sum to the number of times that landmark i was the highest voltage):")
    print(neighborhood)
    print("Row sums:")
    print(np.sum(neighborhood, axis=1))     

    with open(save_data, 'wb') as f:
        pickle.dump(Data, f)
    print(f"added voltage_map to {save_data}")

