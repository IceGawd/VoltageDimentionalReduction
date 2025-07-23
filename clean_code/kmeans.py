"""
Streaming K-Means++ Implementation with FAISS

This module implements a memory-efficient streaming version of the k-means++
clustering algorithm using FAISS for fast nearest neighbor search. It is designed
to handle large datasets that don't fit in memory by processing data in batches.

Key Features:
    - Streaming processing for memory efficiency
    - FAISS-based distance computations for speed
    - Automatic normalization constant estimation
    - Support for normalized and unnormalized vectors
    - Label tracking for supervised applications

Example:
    >>> from kmeans import Streaming_Kmeans
    >>> centroids, counters, label_counts, init_d2, final_d2 = Streaming_Kmeans('data.csv')
    >>> print(f"Found {len(centroids)} clusters")
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional, Counter as CounterType
import numpy.typing as npt

from Utilities import config
from Visualization import generalVisualization
from Utilities.reader import Reader

import faulthandler
from collections import Counter
faulthandler.enable()

# ------------- Streaming KMeans++ --------------
class StreamingKMeansPlusPlus:
    """
    Implements streaming k-means++ centroid selection using FAISS for efficient distance computation.

    This class implements a streaming version of the k-means++ initialization algorithm.
    It processes data in batches and maintains a FAISS index for efficient nearest
    neighbor searches. New centroids are selected probabilistically based on their
    squared distances to existing centroids.

    Attributes:
        d (int): Dimensionality of the input vectors.
        Z (float): Scaling constant for sampling probability (max_dist² - min_dist²).
        shift (float): Minimum squared distance for probability calculation.
        max_centroids (int): Maximum number of centroids to maintain.
        index (faiss.IndexFlatL2): FAISS index for efficient distance computation.

    Note:
        - The algorithm maintains normalized probabilities using Z and shift parameters
        - FAISS index must be of type IndexFlatL2 for correct distance calculations
        - Vectors can optionally be normalized before processing
    """

    def __init__(self, d: int, max_dist2: float, min_dist2: float, index: faiss.IndexFlatL2) -> None:
        """
        Initialize the streaming k-means++ algorithm.

        Args:
            d (int): Dimensionality of the input vectors.
            max_dist2 (float): Maximum squared distance between any two points in initial buffer.
            min_dist2 (float): Minimum squared distance between any two points in initial buffer.
            index (faiss.IndexFlatL2): Pre-initialized FAISS index for distance computations.

        Raises:
            AssertionError: If the provided index is not of type faiss.IndexFlatL2.

        Note:
            - Z is computed as (max_dist² - min_dist²) for probability normalization
            - The shift parameter ensures non-negative distance values
            - max_centroids is loaded from the global config
        """
        self.d = d
        self.Z = max_dist2 - min_dist2  # Normalization constant for sampling probabilities
        self.shift= min_dist2  # Shift to ensure non-negative distances
    
        self.max_centroids = config.params['max_centroids']
        assert type(index) == faiss.IndexFlatL2, "Index must be of type faiss.IndexFlatL2"
        self.index = index
     


    def _compute_distances_squared(self, X, index):
        """
        Computes squared distances from X to nearest centroid in index.

        Args:
            X (np.ndarray): Batch of input vectors.
            index (faiss.IndexFlatL2): FAISS index of centroids.

        Returns:
            np.ndarray: Squared distances for each point in X.
        """
        if self.index is None or self.index.ntotal == 0:
            print("Index is empty, returning infinity distances.")
            return None
        D, I = self.index.search(X, k=1) # D[i, 0] is the squared distance from X[i] to the nearest centroid
        return D[:, 0], I[:, 0]  # Return only the squared distances and indices of nearest centroids   

    def update(self, X_batch):
        """
        Updates centroid list with new vectors selected via probabilistic sampling.

        Args:
            X_batch (np.ndarray): Normalized batch of vectors.
        """
                
        needed_centroids = self.max_centroids - self.index.ntotal
        
        d2,I = self._compute_distances_squared(X_batch, self.index)
        if needed_centroids>0:
            ratio=(d2-self.shift) / self.Z  # Shift distances 
            probs = np.minimum(np.maximum(ratio, 0), 1)  # Ensure probabilities are in [0, 1]
            #print('probs=',np.mean(probs))
            rand_vals = np.random.rand(X_batch.shape[0])
            accept_mask = (rand_vals < probs) & (d2 > 0)
            if np.sum(accept_mask) != 0:
                X=np.stack(X_batch[accept_mask])
                if self.index.ntotal + X.shape[0] > self.max_centroids:
                    X = X[:self.max_centroids - self.index.ntotal]
                if config.params['normalize_vecs']:
                    X = X / np.linalg.norm(X,axis=1, keepdims=True)
                self.index.add(X)  # Add new vectors to the index          
                print(f"\nnumber of centroids: {self.index.ntotal}, max_centroids: {self.max_centroids}")
        
    def get_centroids(self):
        """
        Returns the current list of centroids as a NumPy array.

        Returns:
            np.ndarray: Centroids of shape (num_centroids, d)
        """
        return self.index.reconstruct_n(0, self.index.ntotal)



# ------------------- Streaming_Kmeans----------
def Streaming_Kmeans(filepath: str) -> Tuple[npt.NDArray, npt.NDArray, List[Optional[CounterType]], float, float]:
    """
    Perform streaming k-means++ clustering with FAISS.

    This function implements a complete streaming k-means++ pipeline, including initialization
    and refinement. It processes data in batches to handle large datasets efficiently.

    Algorithm Steps:
        1. Initialize: Read initial buffer to estimate distance parameters
            - Compute pairwise distances using FAISS
            - Determine max_dist² and min_dist² for probability scaling
        2. Select Centroids: Stream data to select initial centroids
            - Use distance-based probabilistic sampling
            - Continue until max_centroids is reached
        3. Refine Centroids: Stream data twice to update centroids
            - First pass: Assign points and update centroids
            - Second pass: Further refine centroids
            - Track label distributions for each centroid

    Args:
        filepath (str): Path to the input data file containing vectors.

    Returns:
        Tuple containing:
            - centroids (np.ndarray): Final centroid vectors (shape: [n_centroids, d])
            - counters (np.ndarray): Number of points assigned to each centroid
            - label_counts (List[Counter]): Distribution of labels for each centroid
            - initial_mean_d2 (float): Initial mean squared distance to centroids
            - mean_d2 (float): Final mean squared distance to centroids

    Note:
        Configuration parameters are read from config.params:
            - 'init_size': Size of initial buffer for parameter estimation
            - 'batch_size': Number of vectors to process per batch
            - 'max_centroids': Maximum number of centroids to select
            - 'normalize_vecs': Whether to normalize input vectors
    """
    reader = Reader(filepath)

    # Step 1: Read initial buffer of vectors for Z estimation
    ######################################

    buffer = []
    d = None
    total_needed = config.params['init_size']
    collected = 0
    for vectors, _ in reader.stream_batches(config.params['batch_size']):
        if d is None:
            d = vectors.shape[1]
        if collected + len(vectors) > total_needed:
            vectors = vectors[:total_needed - collected]
        buffer.append(vectors)
        collected += len(vectors)
        if collected >= total_needed:
            break

    buffer = np.vstack(buffer)
    if config.params['normalize_vecs']:
        buffer= buffer / np.linalg.norm(buffer, axis=1, keepdims=True)

    # Compute all pairwise distances using FAISS and set Z to the maximal distance
    index = faiss.IndexFlatL2(d)  # this should be the one and only place that FAISS in initialized, otherwise there are problems with 
                                  # Initializing openMP more than once
    index.add(buffer)
    D, _ = index.search(buffer, buffer.shape[0])  # D[i, j] is the squared L2 distance from buffer[i] to buffer[j]
    np.fill_diagonal(D, -np.inf)
    max_dist2 = np.max(D[d>0])  # max of square distances
    min_dist2 = np.min(D[D > 0])  # Minimum distance in the buffer (excluding self-distances)
    print(f"\nEstimated max pairwise distance squared, FAISS) = {max_dist2:.4f}")
    print(f" Estimated minimum distance squared = {min_dist2:.4f}")

    # Step 2: Streaming centroid selection
    ######################################
    # For seeding, pick a random vector from the buffer
    centroids = buffer[:10,:]
    index.reset()  # Reset index to ensure it's empty
    index.add(centroids)  # Add the initial centroid to the index
    print(index.ntotal, "vectors in index after adding initial centroid")
    skmeans = StreamingKMeansPlusPlus(d=d, max_dist2=max_dist2, min_dist2=min_dist2,index=index)
    
    for vectors, _ in reader.stream_batches(config.params['batch_size']):
        if config.params['normalize_vecs']:
            vectors=vectors/ np.linalg.norm(vectors, axis=1, keepdims=True)
        
        skmeans.update(vectors)
        if(index.ntotal >= config.params['max_centroids']):
            print(f"Reached maximum number of centroids: {index.ntotal}")
            break

    # Step 3. update centroids using a streaming version of the Kmeans algorithm
    ######################################
    centroids = skmeans.get_centroids()
    counters=np.ones(centroids.shape[0], dtype=np.int32)  # Initialize counters for each centroid
    total_d2 = 0    # Initialize total distance squared to zero
    initial_mean_d2=0
    total_count = 0  # Total number of vectors processed
     # compute for each centroid a label that is the majority of examples that are assigned to it
    # Initialize a list to store the labels assigned to each centroid
    centroid_labels = [ [] for _ in range(centroids.shape[0]) ]

    for vectors, labels in reader.stream_batches(config.params['batch_size']):
        if config.params['normalize_vecs']:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        D, vec_assignments = skmeans._compute_distances_squared(vectors, index)
        # Assign labels to centroids
        for idx, centroid_idx in enumerate(vec_assignments):
            centroid_labels[centroid_idx].append(labels[idx])
        # Count and average the vectors assigned to each centroid using numpy
        unique_values, counts = np.unique(vec_assignments, return_counts=True)
        for i, count in zip(unique_values, counts):
            if count > 0:
                # Update the centroid with the new vectors
                centroids[i] = (centroids[i] * counters[i] + np.sum(vectors[vec_assignments == i], axis=0)) \
                                / (counters[i] + count)
                counters[i] += count
                total_d2 += np.sum(D[vec_assignments == i])  # Sum of squared distances for this centroid
                total_count += count
        mean_d2 = total_d2 / total_count if total_count > 0 else 0
        if initial_mean_d2 == 0:
            initial_mean_d2 = mean_d2
        print('mean d2=', mean_d2, end='')
        skmeans.index.reset()  # Reset index to ensure it's empty
        skmeans.index.add(centroids)  # Add the final centroids to the index

    # Compute majority label for each centroid
    label_counts = []
    for labels_list in centroid_labels:
        if labels_list:
            label_count = Counter(labels_list)
        else:
            label_count = None
        label_counts.append(label_count)
    
    for vectors, labels in reader.stream_batches(config.params['batch_size']):
        if config.params['normalize_vecs']:
            vectors=vectors/ np.linalg.norm(vectors, axis=1, keepdims=True)
        D,vec_assignments= skmeans._compute_distances_squared(vectors, index)
        #count and average the vectors assigned to each centroid using numpy
        unique_values,counts=np.unique(vec_assignments, return_counts=True)
        for i, count in zip(unique_values, counts):
            if count > 0:
                # Update the centroid with the new vectors
                centroids[i] = (centroids[i] * counters[i] + np.sum(vectors[vec_assignments == i], axis=0)) \
                                / (counters[i] + count)
                counters[i] += count
                total_d2 += np.sum(D[vec_assignments == i])  # Sum of squared distances for this centroid
                total_count += count
        mean_d2= total_d2 / total_count if total_count > 0 else 0
        if initial_mean_d2 == 0:
            initial_mean_d2 = mean_d2
        print('mean d2=', mean_d2,end='')
        skmeans.index.reset()  # Reset index to ensure it's empty
        skmeans.index.add(centroids)  # Add the final centroids to the index

    # Close the reader 
    print("\nClosing reader...")
    reader.close()

    return centroids, counters, label_counts, initial_mean_d2, mean_d2


# ------------------- Main ---------------------
from Utilities.set_params import set_params

def main() -> None:
    """
    Main entry point for the streaming k-means++ clustering algorithm.

    This function:
        1. Sets up configuration parameters from command line arguments
        2. Runs the clustering algorithm on the specified input file
        3. Handles test mode with synthetic data
        4. Saves results and generates visualizations if requested

    Configuration (via config.params):
        - test: Boolean, enables test mode with synthetic data
        - file_path: Path to input data file
        - split_char: Delimiter for CSV files
        - normalize_vecs: Whether to normalize vectors
        - max_centroids: Maximum number of centroids
        - init_size: Size of initial buffer
        - batch_size: Number of vectors per batch
        - output: Path for saving results (optional)
        - verbosity: Level of output detail

    In test mode:
        - Uses synthetic 2D random data
        - Validates mean squared distance
        - Generates visualization if successful
    """
    
    set_params()  # Set parameters according to the command line           
    if config.params['test']:
        
        config.params['file_path']= '../../Voltage_Data/synthetic/2drandom10000.csv'
        config.params['split_char']= ','
        config.params['normalize_vecs']= False

        config.params['max_centroids']= 20
        config.params['init_size']= 1000
        config.params['batch_size']= 100
        config.params['output']=None

    centroids,counters,label_counts,inital_mean_d2,mean_d2=Streaming_Kmeans(config.params['file_path'])

    # Finalization and saving
    if config.params['verbosity']>=1:
        print(f"\nNumber of centroids in index after finalization: {centroids.shape[0]}")
        print('Initial mean squared distance:', inital_mean_d2)
        print('Final mean squared distance:', mean_d2)
        if config.params['test']:
            if mean_d2>0.04 or mean_d2<0.035:
                raise ValueError(f"test failed, mean_d2={mean_d2} is outside the range [0.035,0.04]")
            else:
                print('Test Passed')
    if config.params['output'] is not None:
        np.savez(config.params['output'], centroids=centroids, counters=counters, label_counts=label_counts,)
        print(f"Centroids saved to {config.params['output']}")
    else:
        print("No output file specified, centroids not saved.")

    # if 2d test then visualize datapoints, centroids labels
    if config.params['test']:
        generalVisualization.plot_centroids(centroids, counters, label_counts)
    
if __name__ == "__main__":
    main()

