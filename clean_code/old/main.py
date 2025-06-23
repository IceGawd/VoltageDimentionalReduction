import argparse
import numpy as np
import random
import config
from kmeans import Streaming_Kmeans
from problem import Problem
from rg_optimizer import rg_optimizer

def parse_args():
    parser = argparse.ArgumentParser(description="Voltage Dimensional Reduction Pipeline")
    # Add Kmeans parameters (example: --n_clusters, --max_iter, etc.)
    parser.add_argument("file_path", help="Path to a text file of vectors (word + floats)")
 
    parser.add_argument('--num_landmarks', type=int, default=10, help='Number of landmarks to select')
    parser.add_argument("--split_char")
    parser.add_argument("--normalize_vecs", action="store_true", help="normalize vectors to L_2=1 before calculating distances")
    parser.add_argument("--max-centroids", type=int, default=1000, help="Maximum number of centroids")
    parser.add_argument("--init-size", type=int, default=1000, help="Number of points to estimate Z")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for streaming")
    parser.add_argument("--output", type=str, default="streaming_centroids.npy", help="Output .npy file")
    args = parser.parse_args()

    config.params=vars(args)
    filepath = args.file_path
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
    
    return parser.parse_args()

def main():
    args = parse_args()
    # Load data
    # Run Kmeans to get centroids
    kmeans = Streaming_Kmeans(args.file_path)
    centroids = kmeans.cluster_cente
    # Create graph over centroids
    problem = Problem(centroids)
    graph = problem.create_graph()
    # Select landmarks at random
    num_centroids = centroids.shape[0]
    landmark_indices = random.sample(range(num_centroids), args.num_landmarks)
    voltage_functions = []
    for idx in landmark_indices:
        best_r, voltage = rg_optimizer(graph, idx)
        voltage_functions.append(voltage)
    # Store voltage functions as a list of numpy arrays
    # Example: save to disk or use as needed
    np.save('voltage_functions.npy', np.array(voltage_functions))
    print(f"Saved {len(voltage_functions)} voltage functions to voltage_functions.npy")

if __name__ == "__main__":
    main()