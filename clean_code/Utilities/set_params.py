"""
Parameter Configuration Module for Voltage Dimensional Reduction.

This module handles the configuration and validation of parameters for the voltage-based
dimensional reduction pipeline. It provides a command-line interface for setting various
parameters related to data processing, voltage map computation, and visualization.

The configuration is organized into several categories:
- Data file reading: Parameters for input file handling
- Streaming k-means: Parameters for centroid computation
- Voltage maps: Parameters for k-connectivity and resistance
- Intermediate results: Storage locations for pipeline stages
- Landmark selection: Parameters for choosing representative points
- Visualization: Parameters for data visualization
- Miscellaneous: Debug levels and test modes

Example:
    To use this module with default parameters:
    >>> from Utilities.set_params import set_params
    >>> set_params()  # Uses defaults for MNIST dataset
    
    To override defaults via command line:
    $ python script.py --k 15 --normalize_vecs --batch-size 2000
"""

import argparse
from Utilities import config

def set_params():
    """
    Set and validate configuration parameters for the voltage dimensional reduction pipeline.
    
    This function initializes and processes command-line arguments, setting up all necessary
    parameters for the pipeline. It performs validation on the parameters and stores them
    in the global config object.
    
    Configuration Categories:
        Data Reading:
            - file_path: Path to input data file
            - split_char: Delimiter for parsing input vectors
            
        Streaming K-means:
            - normalize_vecs: Whether to normalize vectors to unit length
            - max-centroids: Upper limit on number of centroids
            - init-size: Sample size for Z estimation
            - batch-size: Size of streaming batches
            
        Voltage Maps:
            - k: Connectivity parameter for k-NN graph
            - r: Ground resistance value
            - sigma: RBF kernel parameter
            
        Results Storage:
            - save_data: Path for intermediate results pickle file
            
        Landmark Selection:
            - NoOfLandmarks: Number of landmarks to select
            - DepthOfLandmarkSearch: Search depth for landmark selection
    
    Returns:
        None. Parameters are stored in config.params dictionary.
        
    Raises:
        ValueError: If any numerical parameters are invalid (<=0)
        
    Note:
        - Default values are optimized for the MNIST dataset
        - Use --test flag for quick debugging with reduced parameters
        - Verbosity levels: 0 (silent), 1 (normal), 2 (verbose)
    """
    parser = argparse.ArgumentParser(description="Set parameters for the streaming centroids algorithm.")
    ## defaults are set for the MNIST dataset

    ## data file reading parameters
    parser.add_argument("file_path", nargs='?', default='../../Voltage_Data/mnist/mnist.csv', help="Path to a text file of vectors (word + floats)")
    parser.add_argument("--split_char", type=str, default=",", help="Character to split input vectors")
    
    parser.add_argument("--output_path", type=str, default='', help="Path to the output filtered text file")

    # parameters for streaming k-means
    parser.add_argument("--normalize_vecs", action="store_true", help="normalize vectors to L_2=1 before calculating distances")
    parser.add_argument("--max_centroids", type=int, default=1000, help="Maximum number of centroids")
    parser.add_argument("--init_size", type=int, default=1000, help="Number of points to estimate Z")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for streaming")
    parser.add_argument("--kmeans_alpha", type=float, default=0.1, 
                        help="defines fraction of new centroid that comes from the average of assigned vectors (0.0 to 1.0)")
    parser.add_argument("--equalize_centroids", action="store_true", help="Equalize centroids by removing small ones and splitting large ones")

    # parameters for computing voltage maps
    parser.add_argument("--k", type=int, default=10, help="k-connectivity for the k-nearest neighbor graph")
    parser.add_argument("--r", type=int, default=1, help="resistance to ground")
    parser.add_argument("--sigma", type=float, default=None, help="Sigma value for RBF weighting (default: auto)")

    # parameters for storing intermediate results
    # Change from previous versions, all of the intermediate results are saved in a single pickle file
    # Each program in the pipeline (main, select_landmarks, xgb, visualization in jupyter notebook)reads this file and update it with its results
    parser.add_argument("--save_data", type=str, default="../../Voltage_Temp/Results/mnist/saved_data.pkl", help="Path to the output pickle file for saved data")

    # parameters for landmark selection
    parser.add_argument("--NoOfLandmarks", type=int, default=10, help="Number of landmarks to select for the voltage map")
    parser.add_argument("--DepthOfLandmarkSearch", type=int, default=100, help="Depth of landmark search")
 
    # parameters for visualization
    parser.add_argument("--plot_dir", type=str, default='', help="Directory to save output plots")
    parser.add_argument("--num_labels", type=int, default=10, help="Number of labels for visualization")
    parser.add_argument("--ratio_threshold", type=float, default=0.5, help="Threshold for ratio of the most common label to total count")
    parser.add_argument("--scatter_element", type=str, default='digit', help="Element type for scatter plot")
    parser.add_argument("--landmarkSize", type=int, default=3, help="Size of landmark markers in scatter plot")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha value for scatter plot")
    parser.add_argument("--percent_size", type=float, default=0.02, help="Percentage size for scatter plot")
    parser.add_argument("--no_log_transform", action="store_true", help="Disable log transformation of voltages for visualization")
    
    # parameters for filtering
    parser.add_argument("--indices", type=list, default=[],nargs='+', help="List of indices of voltage maps accodring to which we filter the dataset")
    parser.add_argument("--no_filter", action="store_true", help="If set, disables filtering by indices")
    parser.add_argument("--filter_partition", action="store_true", help="If set, filter the dataset into one output file per landmark")
    
    # misc parameters
    parser.add_argument("--verbosity", type=int, default=1, help="Verbosity level of debug printouts (0: silent, 1: normal, 2: verbose)")
    parser.add_argument("--test", action="store_true", help="Run in test mode with reduced parameters for quick debugging")
    args = parser.parse_args()

    config.params=vars(args)
    if config.params['verbosity']>=2:
        print("Configuration parameters:")
        for key, value in config.params.items():
            if type(value) is str:
                value = re.sub(r'\s+', ' ', value)
                value=f"'{value}'"
        
    # Validate input parameters
    if args.max_centroids <= 0:
        raise ValueError("max-centroids must be a positive integer.")
    if args.init_size <= 0:
        raise ValueError("init-size must be a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be a positive integer.")
    if args.normalize_vecs:
        print("Normalizing vectors to L2=1 before distance calculations.")
    else:
        print("Using raw vectors without normalization for distance calculations.")
