import numpy as np
import faiss
import os
import argparse
import config
# ------------------- Reader -------------------
class ParseException(Exception):
    pass

def readvec(file):
    line = file.readline()
    if not line:
        return None
    parts = line.strip().split()
    if len(parts) < 2:
        raise ParseException(parts)
    try:
        vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        return vec
    except ValueError:
        return None  # Skip lines with bad floats

    except ValueError:
        raise ParseException(parts)

class Reader:
    """
    Reads a text file containing vectors line-by-line and yields batches of vectors.

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
        Generator that yields batches of vectors as NumPy arrays.

        Args:
            batch_size (int): Number of vectors to include in each batch.

        Yields:
            np.ndarray: A batch of shape (batch_size, vector_dim)
        """
        while True:
            vectors = []
            for _ in range(batch_size):
                vec=readvec(self.file)
                if not vec is None:
                    vectors.append(vec)
                    self.counter += 1
                    if self.counter % 1000 == 0:
                        print(f"\rRead {self.counter} vectors", end='', flush=True)
            if not vectors:
                break
            yield np.stack(vectors)

    def close(self):
        """
        Closes the file handle.
        """
        self.file.close()


# ------------- Streaming KMeans++ --------------
class StreamingKMeansPlusPlusFAISS:
    """
    Implements streaming k-means++ centroid selection using FAISS for efficient distance computation.

    Attributes:
        d (int): Dimensionality of vectors.
        Z (float): Scaling constant for sampling probability.
        max_centroids (int): Maximum number of centroids to retain.
        centroids (List[np.ndarray]): List of current centroids.
    """

    def __init__(self, d, Z):
        """
        Initializes the streaming k-means++ class.

        Args:
            d (int): Vector dimensionality.
            Z (float): Normalization constant for sampling.
            max_centroids (int): Maximum number of centroids to store.
        """
        self.centroids = []
        self.d = d
        self.Z = Z
        self.max_centroids = config.params.max_centroids

    def _build_faiss_index(self):
        """
        Builds a FAISS index over current centroids.

        Returns:
            faiss.IndexFlatL2: FAISS index with current centroids or None if empty.
        """
        if not self.centroids:
            return None
        stack_centroids=np.stack(self.centroids)
        if config.params.normalize_dist:
            faiss.normalize_L2(stack_centroids)
        index = faiss.IndexFlatL2(self.d)
        index.add(stack_centroids)
        return index

    def _compute_distances_squared(self, X, index):
        """
        Computes squared distances from X to nearest centroid in index.

        Args:
            X (np.ndarray): Batch of input vectors.
            index (faiss.IndexFlatL2): FAISS index of centroids.

        Returns:
            np.ndarray: Squared distances for each point in X.
        """
        if index is None or index.ntotal == 0:
            return np.full(X.shape[0], np.inf, dtype=np.float32)
        D, _ = index.search(X, 1)
        return D[:, 0]

    def update(self, X_batch):
        """
        Updates centroid list with new vectors selected via probabilistic sampling.

        Args:
            X_batch (np.ndarray): Normalized batch of vectors.
        """
        index = self._build_faiss_index()
        d2 = self._compute_distances_squared(X_batch, index)
        probs = d2 / self.Z
        rand_vals = np.random.rand(X_batch.shape[0])
        accept_mask = (rand_vals < probs) & (d2 > 0)

        for x in X_batch[accept_mask]:
            if config.params.normalize_dist:
                x = x / np.linalg.norm(x)
            if len(self.centroids) < self.max_centroids:
                self.centroids.append(x.copy())
            else:
                break

    def get_centroids(self):
        """
        Returns the current list of centroids as a NumPy array.

        Returns:
            np.ndarray: Centroids of shape (num_centroids, d)
        """
        centroids = np.stack(self.centroids) if self.centroids else np.empty((0, self.d), dtype=np.float32)
        if config.params.normalize_dist and centroids.shape[0] > 0:
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            centroids = centroids / np.maximum(norms, 1e-10)
        return centroids

# ------------------- Streaming_Kmeans----------
def Streaming_Kmeans(filepath):

    """
    Main function to perform streaming k-means++ with FAISS.

    Steps:
        1. Estimate normalization constant Z from an initial buffer.
        2. Select centroids incrementally using streaming batches.
        3. Save final centroids to a .npy file.

    parameters are passed through config.params, see listing of parameters in argparse section.
    """
    reader = Reader(filepath)

    # Step 1: Read initial buffer of vectors for Z estimation
    buffer = []
    d = None
    total_needed = config.params.init_size
    collected = 0
    for batch in reader.stream_batches(config.params.batch_size):
        if d is None:
            d = batch.shape[1]
        if collected + len(batch) > total_needed:
            batch = batch[:total_needed - collected]
        buffer.append(batch)
        collected += len(batch)
        if collected >= total_needed:
            break

    buffer = np.vstack(buffer)
    if config.params.normalize_dist:
        faiss.normalize_L2(buffer)

    # Compute all pairwise distances using FAISS and set Z to the maximal distance
    index = faiss.IndexFlatL2(d)
    index.add(buffer)
    D, _ = index.search(buffer, buffer.shape[0])  # D[i, j] is the squared L2 distance from buffer[i] to buffer[j]
    np.fill_diagonal(D, -np.inf)
    Z = np.sqrt(np.max(D))  # Take sqrt because FAISS returns squared distances
    print(f"\nEstimated Z (max pairwise distance, FAISS) = {Z:.4f}")
    print(f"minimum distance in buffer = {np.sqrt(np.min(D[D > 0])):.4f}")

    # Print 0.1% and 99% percentiles of the non-diagonal values in D
    D_flat = D[D > 0]  # Exclude zeros (self-distances)
    p01 = np.sqrt(np.percentile(D_flat, 0.1))
    p99 = np.sqrt(np.percentile(D_flat, 99))
    print(f"0.1% percentile distance in buffer = {p01:.4f}")
    print(f"99% percentile distance in buffer = {p99:.4f}")

    # Plot histogram of pairwise distances in D_flat and overlay with nearest neighbor distances
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    # Histogram of all pairwise distances (normalized to be a distribution)
    plt.hist(np.sqrt(D_flat), bins=100, color='blue', alpha=0.5, label='All Pairwise Distances', density=True)
    # Histogram of nearest neighbor distances (normalized to be a distribution)
    D_no_diag = D.copy()
    np.fill_diagonal(D_no_diag, np.inf)
    nn1_distances = np.sqrt(np.partition(D_no_diag, 1, axis=1)[:, 1])  # 1st nearest neighbor (not self)
    nn2_distances = np.sqrt(np.partition(D_no_diag, 2, axis=1)[:, 2])  # 2nd nearest neighbor
    nn3_distances = np.sqrt(np.partition(D_no_diag, 3, axis=1)[:, 3])  # 3rd nearest neighbor
    nn10_distances = np.sqrt(np.partition(D_no_diag, 10, axis=1)[:, 10])  # 10th nearest neighbor
    plt.hist(nn1_distances, bins=100, color='red', alpha=0.5, label='1st Nearest Neighbor', density=True)
    plt.hist(nn2_distances, bins=100, color='green', alpha=0.5, label='2nd Nearest Neighbor', density=True)
    plt.hist(nn3_distances, bins=100, color='orange', alpha=0.5, label='3rd Nearest Neighbor', density=True)
    plt.hist(nn10_distances, bins=100, color='purple', alpha=0.5, label='10th Nearest Neighbor', density=True)
    plt.title('Histogram of Pairwise and k-th Nearest Neighbor Distances in Buffer')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Step 2: Streaming centroid selection
    # For seeding, pick a random vector from the buffer
    centroid = buffer[np.random.randint(len(buffer))]
    skmeans = StreamingKMeansPlusPlusFAISS(d=d, Z=Z)
    if config.params.normalize_dist:
        centroid = centroid.reshape(1, -1)
        faiss.normalize_L2(centroid)
        centroid = centroid[0]
    skmeans.centroids.append(centroid)  # seed with the first point

    for batch in reader.stream_batches(config.params.batch_size):
        if config.params.normalize_dist:
            faiss.normalize_L2(batch)
        skmeans.update(batch)

    reader.close()

    return skmeans


# ------------------- Main ---------------------
def main():
    parser = argparse.ArgumentParser(description="Streaming k-means++ with FAISS")
    parser.add_argument("file_path", help="Path to a text file of vectors (word + floats)")
    parser.add_argument("--normalize_dist", action="store_true",help="normalize vectors to L_2=1 before calculating distances")
    parser.add_argument("--max-centroids", type=int, default=1000, help="Maximum number of centroids")
    parser.add_argument("--init-size", type=int, default=1000, help="Number of points to estimate Z")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for streaming")
    parser.add_argument("--output", type=str, default="streaming_centroids.npy", help="Output .npy file")
    args = parser.parse_args()

    config.params=args
    filepath = args.file_path
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file {filepath} does not exist.")
    if not filepath.endswith('.txt'):
        raise ValueError(f"Output directory {os.path.dirname(args.output)} does not exist.")
    if args.max_centroids <= 0:
        raise ValueError("max-centroids must be a positive integer.")
    if args.init_size <= 0:
        raise ValueError("init-size must be a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be a positive integer.")
    if args.normalize_dist:
        print("Normalizing vectors to L2=1 before distance calculations.")
    else:
        print("Using raw vectors without normalization for distance calculations.")
    
    skmeans=Streaming_Kmeans(filepath)
 
    centroids = skmeans.get_centroids()
    print(f"\nFinal number of centroids: {centroids.shape[0]}")
    np.save(args.output, centroids)
    print(f"Centroids saved to {args.output}")


if __name__ == "__main__":
    main()

