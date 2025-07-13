import argparse
from Utilities import config

def set_params():
    parser = argparse.ArgumentParser(description="Set parameters for the streaming centroids algorithm.")
    parser.add_argument("file_path", nargs='?', default=None, help="Path to a text file of vectors (word + floats)")
    parser.add_argument("--split_char")
    parser.add_argument("--normalize_vecs", action="store_true", help="normalize vectors to L_2=1 before calculating distances")
    parser.add_argument("--max-centroids", type=int, default=1000, help="Maximum number of centroids")
    parser.add_argument("--init-size", type=int, default=1000, help="Number of points to estimate Z")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for streaming")
    parser.add_argument("--output", type=str, default="streaming_centroids.npy", help="Output .npy file")
    parser.add_argument("--verbosity", type=int, default=1, help="Verbosity level (0: silent, 1: normal, 2: verbose)")
    parser.add_argument("--test", action="store_true", help="run in self-test mode")
    parser.add_argument("--k", type=int, default=10, help="k-connectivity for the k-nearest neighbor graph")
    parser.add_argument("--sigma", type=float, default=None, help="Sigma value for RBF weighting (default: auto)")
    parser.add_argument("--NoOfLandmarks", type=int, default=10, help="Number of landmarks to select for the voltage map")
    args = parser.parse_args()

    config.params=vars(args)
    if config.params['verbosity']>=2:
        print("Configuration parameters:")
        for key, value in config.params.items():
            if type(value) is str:
                value = re.sub(r'\s+', ' ', value)
                value=f"'{value}'"
        
    # Validate input parameters
    filepath = args.file_path
    if not config.params['test']:
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Input file {filepath} does not exist.")
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
