import argparse
from Utilities import config

def set_params():
    parser = argparse.ArgumentParser(description="Set parameters for the streaming centroids algorithm.")
    ## defaults are set for the MNIST dataset

    ## data file reading parameters
    parser.add_argument("file_path", nargs='?', default='../../Voltage_Data/mnist/mnist.csv', help="Path to a text file of vectors (word + floats)")
    parser.add_argument("--split_char", type=str, default=",", help="Character to split input vectors")
    
    # parameters for streaming k-means
    parser.add_argument("--normalize_vecs", action="store_true", help="normalize vectors to L_2=1 before calculating distances")
    parser.add_argument("--max-centroids", type=int, default=1000, help="Maximum number of centroids")
    parser.add_argument("--init-size", type=int, default=5000, help="Number of points to estimate Z")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for streaming")

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
