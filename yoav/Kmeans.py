import numpy as np
import faiss
import os
import argparse

# ------------------- Reader -------------------
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
                line = self.file.readline()
                if not line:
                    break
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    vectors.append(vec)
                    self.counter += 1
                    if self.counter % 1000 == 0:
                        print(f"\rRead {self.counter} vectors", end='', flush=True)
                except ValueError:
                    continue  # Skip lines with bad floats
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

    def __init__(self, d, Z, max_centroids):
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
        self.max_centroids = max_centroids

    def _build_faiss_index(self):
        """
        Builds a FAISS index over current centroids.

        Returns:
            faiss.IndexFlatL2: FAISS index with current centroids or None if empty.
        """
        if not self.centroids:
            return None
        index = faiss.IndexFlatL2(self.d)
        index.add(np.stack(self.centroids))
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
        return np.stack(self.centroids) if self.centroids else np.empty((0, self.d), dtype=np.float32)


# ------------------- Main ---------------------
def main():
    """
    Main function to perform streaming k-means++ with FAISS.

    Steps:
        1. Estimate normalization constant Z from an initial buffer.
        2. Select centroids incrementally using streaming batches.
        3. Save final centroids to a .npy file.

    Command-line Arguments:
        file_path (str): Path to input file with vectors.
        --max-centroids (int): Max number of centroids (default=1000).
        --init-size (int): Number of initial points to estimate Z (default=10000).
        --batch-size (int): Size of streaming batches (default=1000).
        --output (str): Output file for centroids (.npy format).
    """
    parser = argparse.ArgumentParser(description="Streaming k-means++ with FAISS")
    parser.add_argument("file_path", help="Path to a text file of vectors (word + floats)")
    parser.add_argument("--max-centroids", type=int, default=1000, help="Maximum number of centroids")
    parser.add_argument("--init-size", type=int, default=10000, help="Number of points to estimate Z")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for streaming")
    parser.add_argument("--output", type=str, default="streaming_centroids.npy", help="Output .npy file")
    args = parser.parse_args()

    reader = Reader(args.file_path)

    # Step 1: Estimate Z from an initial buffer
    buffer = []
    centroid = None
    d = None

    for batch in reader.stream_batches(args.batch_size):
        if centroid is None:
            centroid = batch[np.random.randint(len(batch))]
            d = batch.shape[1]
        buffer.append(batch)
        if sum(len(b) for b in buffer) >= args.init_size:
            break

    buffer = np.vstack(buffer)
    buffer /= np.linalg.norm(buffer, axis=1, keepdims=True)

    centroid = centroid / np.linalg.norm(centroid)
    index = faiss.IndexFlatL2(d)
    index.add(centroid.reshape(1, -1))
    D, _ = index.search(buffer, 1)
    print(D)
    Z = np.percentile(D[:, 0], 99)
    print(f"\nEstimated Z = {Z:.4f}")

    # Step 2: Streaming centroid selection
    skmeans = StreamingKMeansPlusPlusFAISS(d=d, Z=Z, max_centroids=args.max_centroids)
    skmeans.centroids.append(centroid)  # seed with the first point

    for batch in reader.stream_batches(args.batch_size):
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        batch = batch / np.maximum(norms, 1e-10)  # normalize, avoid divide-by-zero
        skmeans.update(batch)

    reader.close()

    centroids = skmeans.get_centroids()
    print(f"\nFinal number of centroids: {centroids.shape[0]}")
    np.save(args.output, centroids)
    print(f"Centroids saved to {args.output}")


if __name__ == "__main__":
    main()

