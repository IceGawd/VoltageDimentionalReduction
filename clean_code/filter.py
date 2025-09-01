"""
External Memory File Shuffling Utility

This module provides to efficienty filter a a voltages  file according to the voltages. 
It is given as input a list of indices and accepts a line if the largest voltage in the line is in the list of indices.

The implementation uses streaming of the input and output so as to minimize memory usage.

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
import pickle

def filter_voltages(input_path: str, output_path: str, voltages_path: str,indices: list[int],BatchSize:int = 10000) -> None:
    """
    Filters lines from a large text file based on the largest voltage in each line.

    Args:
        input_path (str): Path to the input text file.
        output_path (str): Path to the output text file.
        voltages_path (str): Path to pkl file containing voltage data.
        indices (list[int]): List of indices to filter lines by the largest voltage.

    Returns:
        None
    """

    with open(voltages_path, "rb") as f:
        data = pickle.load(f)
    centroids, voltage_map = data['centroids'], data['voltage_map']


    from xgb import embed_voltage_features
    from Utilities.reader import Reader

    reader = Reader(input_path)
    counter=0
    outfile = open(output_path, 'w')
    for vectors,labels in reader.stream_batches(BatchSize):
        # Assuming vectors is a 2D numpy array where each row is a data point
        # and labels is a 1D numpy array of corresponding labels.
        features = embed_voltage_features(vectors, centroids, voltage_map)

        # Check if the largest voltage in the features is in the specified indices
        for i, feature in enumerate(features):
            if np.argmax(feature) in indices:
                # Write the corresponding line to the output file
                String=np.array2string(vectors[i,:], separator=",", max_line_width=np.inf)[1:-1].replace(" ", "")
                #print(String[:30])
                outfile.write(f"{labels[i]},{String}\n")
                counter+=1

    outfile.close

    return counter

def main() -> None:
    """
    Main function to parse command-line arguments and filter a large input file.
    """
    Usage=""""
        Usage:
        python filter.py --input_path <input_file> --output_path <output_file> --voltages_path <voltages_file> --indices <index1,index2,...> 
        Where:
        --input_path: Path to the input text file.
        --output_path: Path to the output text file.
        --save_data: Path to pkl file containing voltage data.
        --indices: Comma-separated list of indices to filter lines by the largest voltage
    """

    from Utilities.set_params import set_params 
    from Utilities import config
    set_params()
    
    # Check required parameters
    required_params = ['file_path', 'output_path', 'save_data', 'indices']
    for param in required_params:
        if param not in config.params:
            print(Usage)
            raise ValueError(f"Missing required parameter: {param}")

    counter = filter_voltages(
        input_path=config.params['file_path'],
        output_path=config.params['output_path'],
        voltages_path=config.params['save_data'],
        indices=config.params['indices'],
        BatchSize=config.params['batch_size']
    )
    print(f"Filtered {counter} lines to {config.params['output_path']}")
if __name__ == "__main__":
    main()
