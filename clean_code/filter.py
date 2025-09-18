"""
Data Filtering

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
from Utilities.reader import Reader, Writer

def stream_voltages(reader: Reader, voltage_map, centroids,BatchSize:int = 10000):
    """
    reads lines from a large text file an transforms them to voltage features using precomputed centroids and voltage map.

    Args:
        input_path (str): Path to the input text file.
        data_path (str): Path to pkl file containing voltage data.
        BatchSize (int): Number of lines to process in each batch.
    Yields:
        label: The label of the data point.
        vector: The raw data point.
        feature: The interpolated feature vector of the data point.
    """

    from xgb import embed_voltage_features
    

    for vectors,other in reader.stream_batches(BatchSize):
        # Assuming vectors is a 2D numpy array where each row is a data point
        # and labels is a 1D numpy array of corresponding labels.
        try:
            features = embed_voltage_features(vectors, centroids, voltage_map)
        except Exception as e:      
            print(f"Error in embed_voltage_features: {e}")
            break

        for i, feature in enumerate(features):
            yield other[i], vectors[i], features[i]
    print(f"Total lines processed in stream_voltages: {reader.get_counter()}")

def filter_voltages(input_path: str, output_path: str, data_path: str, indices: list[int], BatchSize: int = 10000) -> int:
    """
    Filters lines from a large text file based on the largest voltage in each line. 
    Args:
        input_path (str): Path to the input text file.
        output_path (str): Path to the output text file.
        data_path (str): Path to pkl file containing voltage data.
        indices (list[int]): List of indices to filter lines by the largest voltage.
    """
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    centroids, voltage_map = data['centroids'], data['voltage_map']

    n_landmarks = len(voltage_map)
    
    counter=0
    reader = Reader(input_path)
    if reader is None:
        return
    stream = stream_voltages(reader, voltage_map, centroids, BatchSize)
    from Utilities import config
    import pandas as pd
    if not config.params['filter_partition']:
        writer=Writer(output_path,reader)
        writer.write_header()
        collected_rows=pd.DataFrame(columns=reader.df_sample.columns)
        for other, vector, feature in stream:
            if np.argmax(feature) in indices:
               #write out the row
               writer.write_row(vector, other)

               counter+=1
               if counter % 10000 ==0:
                   print(f"Filtered {counter} lines so far",end='\r')
        writer.write_batch(collected_rows)
    else:
        # partition the output into multiple files, one for each index in indices
        print(f"Partitioning output into {n_landmarks} files")
        # define output_indexed to be the output path without the suffix and suffix to be the suffix


        import os

        # construct output filename from output_path
        if output_path.endswith('.csv'):
            output_stem=output_path[:-4]
        else:
            raise  ValueError("output_path must end with .csv") 
        
        
        print(f"output_path: {output_path}, output_stem={output_stem}")

        # create output files
        writers = {}
        for index in range(n_landmarks):
            file_path = f"{output_stem}_{index}.csv"
            print(f"Creating file: {file_path}")
            writers[index] = Writer(file_path, reader)

        for writer in writers.values():
            writer.write_header()
        for label, vector, feature in stream:
            index = np.argmax(feature)
            # add row to writers
            writers[index].write_row(vector, label)
            counter+=1
            if counter % 10000 ==0:
                print(f"Filtered {counter} lines so far",end='\r')

            
        
    return counter


""" count_neighborhoods is a function that generates a matrix of size num_landmarks x num_landmarks
where the entry i,j is the count of the number of times that landmark i has the largest voltage
and landmark j has the second largest voltage"""
def count_neighborhoods(input_path: str, voltage_map: np.ndarray, centroids: np.ndarray, BatchSize:int = 10000) -> np.ndarray:
    """
    Counts the occurrences of each pair of landmarks being the first and second highest voltages.

    Args:
        input_path (str): Path to the input text file.
        data_path (str): Path to pkl file containing voltage data.
        BatchSize (int): Number of lines to process in each batch.

    Returns:
        np.ndarray: A 2D array where entry (i, j) is the count of times landmark i is the highest voltage
                    and landmark j is the second highest voltage.

                    It follows the same design pattern as filter_voltages and uses stream_voltages
    """
    reader = Reader(input_path)
    if reader is None:
        return
    stream = stream_voltages(reader, voltage_map, centroids, BatchSize)

    num_landmarks = len(voltage_map)

    # Initialize the count matrix
    count_matrix = np.zeros((num_landmarks, num_landmarks), dtype=int)

    for _, _, feature in stream:
        # Get indices of the top two landmarks
        top_two_indices = np.argsort(feature)[-2:][::-1]  # Indices of the two largest values
        first, second = top_two_indices
        count_matrix[first, second] += 1

    return count_matrix

def main() -> None:
    """
    Main function to parse command-line arguments and filter a large input file.
    """
    Usage=""""
        Usage:
        python filter.py --input_path <input_file> --output_path <output_file> --data_path <voltages_file> --indices <index1,index2,...> 
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
        data_path=config.params['save_data'],
        indices=config.params['indices'],
        BatchSize=config.params['batch_size']
    )
    print(f"Filtered {counter} lines to {config.params['output_path']}")
if __name__ == "__main__":
    main()
