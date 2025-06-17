import numpy as np
import faiss
import os
import re
import argparse
import config

import faulthandler
from collections import Counter
faulthandler.enable()

# ------------------- Reader -------------------
class ParseException(Exception):
    pass

def readvec(file):
    line = file.readline()
    if not line:
        return None, None
    split_char = config.params['split_char']
    if split_char == '' or split_char is None:
        parts = line.strip().split()  # default: split on any whitespace
    else:
        parts = line.strip().split(split_char)
    if len(parts) < 2:
        print(line)
        print('no of parts=', len(parts))
        raise ParseException(parts)
    try:
        label = parts[0]
        vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        return label, vec
    except ValueError:
        return None, None  # Skip lines with bad floats

    except ValueError:
        raise ParseException(parts)

class Reader:
    """
    Reads a text file containing vectors line-by-line and yields batches of vectors and labels.

    Each line in the file should be in the format:
        word val1 val2 val3 ...

    Attributes:
        file (TextIO): Opened file handle.
        counter (int): Number of vectors successfully read.
    """

    def __init__(self, file_path):
        """
        Initializes the Reader.

        Args:
            file_path (str): Path to the input text file.
        """
        self.file = open(file_path, 'r', encoding='utf-8')
        self.counter = 0

    def stream_batches(self, batch_size):
        """
        Generator that yields batches of vectors and labels as NumPy arrays.

        Args:
            batch_size (int): Number of vectors to include in each batch.

        Yields:
            tuple: (np.ndarray of shape (batch_size, vector_dim), np.ndarray of shape (batch_size,))
        """
        while True:
            vectors = []
            labels = []
            for _ in range(batch_size):
                label, vec = readvec(self.file)
                if vec is not None:
                    vectors.append(vec)
                    labels.append(label)
                    self.counter += 1
                    if self.counter % config.params['batch_size'] == 0:
                        print(f"\rRead {self.counter} vectors", end='', flush=True)
            if not vectors:
                break
            yield np.stack(vectors), np.array(labels)

    def close(self):
        """
        Closes the file handle.
        """
        self.file.close()


# ------------- Streaming KMeans++ --------------
class StreamingKMeansPlusPlus:
    """
    Implements streaming k-means++ centroid selection using FAISS for efficient distance computation.

    Attributes:
        d (int): Dimensionality of vectors.
        Z (float): Scaling constant for sampling probability.
        max_centroids (int): Maximum number of centroids to retain.
    """

    def __init__(self, d, max_dist2,min_dist2,index):
        """
        Initializes the streaming k-means++ class.

        Args:
            d (int): Vector dimensionality.
            Z (float): Normalization constant for sampling.
            max_centroids (int): Maximum number of centroids to store.
            index (faiss.IndexFlatL2): FAISS index for efficient distance computation.
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
def Streaming_Kmeans(filepath):

    """
    Main function to perform streaming k-means++ with FAISS.

    Steps:
        1. Estimate normalization constant Z from an initial buffer.
        2. Select centroids incrementally using streaming batches.
        3. update centroids using a streaming version of the Kmeans algorithm
        4. Save final centroids to a .npy file.

    parameters are passed through config.params, see listing of parameters in argparse section.
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
     # why are you not doing anything with labels?
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
    majority_labels = []
    for labels_list in centroid_labels:
        if labels_list:
            majority_label = Counter(labels_list).most_common(1)[0][0]
        else:
            majority_label = None
        majority_labels.append(majority_label)
    
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

    return centroids, counters, majority_labels, initial_mean_d2, mean_d2


# ------------------- Main ---------------------
def main():
    parser = argparse.ArgumentParser(description="Streaming k-means++ with FAISS")
    parser.add_argument("file_path", help="Path to a text file of vectors (word + floats)")
    parser.add_argument("--split_char")
    parser.add_argument("--normalize_vecs", action="store_true", help="normalize vectors to L_2=1 before calculating distances")
    parser.add_argument("--max-centroids", type=int, default=1000, help="Maximum number of centroids")
    parser.add_argument("--init-size", type=int, default=1000, help="Number of points to estimate Z")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for streaming")
    parser.add_argument("--output", type=str, default="streaming_centroids.npy", help="Output .npy file")
    parser.add_argument("--verbosity", type=int, default=1, help="Verbosity level (0: silent, 1: normal, 2: verbose)")
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
    
    centroids,counters,majority_labels,inital_mean_d2,mean_d2=Streaming_Kmeans(filepath)
     
    # Finalization and saving
    if config.params['verbosity']>=1:
        print(f"\nNumber of centroids in index after finalization: {centroids.shape[0]}")
        print('Initial mean squared distance:', inital_mean_d2)
        print('Final mean squared distance:', mean_d2)
    # Save the final centroids to a .npy file
    if config.params['output'] is not None:
        np.savez(config.params['output'], centroids=centroids, counters=counters, majority_labels=majority_labels,)
        print(f"Centroids saved to {config.params['output']}")
    else:
        print("No output file specified, centroids not saved.")

# if 2d then visualize datapoints, centroids labels and voronoy diagram
    if centroids.shape[1] == 2:
        import visualization
        visualization.plot_centroids(centroids, counters, majority_labels)


if __name__ == "__main__":
    main()

