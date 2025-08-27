"""
Data Filtering

This module provides to efficienty filter a a voltages  file according to the voltages. 
It is given as input a list of indices and accepts a line if the largest voltage in the line is in the list of indices.

The implementation uses streaming of the input and output so as to minimize memory usage.
"""

import numpy as np
import pickle

def stream_voltages(input_path: str, voltage_map, centroids,BatchSize:int = 10000):
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
    from Utilities.reader import Reader

    reader = Reader(input_path)

    for vectors,labels in reader.stream_batches(BatchSize):
        # Assuming vectors is a 2D numpy array where each row is a data point
        # and labels is a 1D numpy array of corresponding labels.
        features = embed_voltage_features(vectors, centroids, voltage_map)

        for i, feature in enumerate(features):
            yield labels[i], vectors[i], features[i]
    print(f"Total lines processed in stream_voltags: {reader.get_counter()}")

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

    
    counter=0
    stream = stream_voltages(input_path, voltage_map, centroids, BatchSize)
    outfile = open(output_path, 'w')

    for label, vector, feature in stream:
        if np.argmax(feature) in indices:
            # Write the corresponding line to the output file
            String=np.array2string(vector, separator=",", max_line_width=np.inf)[1:-1].replace(" ", "")
            outfile.write(f"{label},{String}\n")
            counter+=1
    outfile.close
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
    stream = stream_voltages(input_path, voltage_map, centroids, BatchSize)

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
