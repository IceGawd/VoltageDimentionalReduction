
import numpy as np
import os
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import kagglehub

# -------------------- Reader --------------------
class Reader:
    def __init__(self, path):
        self.file_path = self._find_glove_file(path)
        self.file = open(self.file_path, 'r', encoding='utf-8')
        self.counter = 0

    def _find_glove_file(self, path):
        for fname in os.listdir(path):
            if fname.endswith('.txt') and '300d' in fname:
                return os.path.join(path, fname)
        raise FileNotFoundError("Could not find GloVe 300d file in the path.")

    def stream_batches(self, batch_size):
        while True:
            L = []
            words = []
            for _ in range(batch_size):
                line = self.file.readline()
                if line == '':
                    break
                parts = line.strip().split()
                if len(parts) != 301:
                    continue  # Malformed line
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                words.append(word)
                L.append(vec)
                self.counter += 1
                if self.counter % 1000 == 0:
                    print(f"\rRead {self.counter} vectors", end='', flush=True)
            if not L:
                break
            yield np.array(words), np.stack(L)

    def close(self):
        self.file.close()

# -------------------- Main --------------------
def main():
    # Parameters
    k = 1000
    init_size = 10000
    batch_size = 1000

    # Download the GloVe dataset
    # path = kagglehub.dataset_download("thanakomsn/glove6b300dtxt")
    path='shuffled_output.txt'
    reader = Reader(path)

    # Step 1: Initialize from first N vectors using k-means++

    words_init, X_init = next(reader.stream_batches(init_size))
    X_init = normalize(X_init, axis=1)

    print("\nFitting initial model with KMeans++")
    kmeans = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1,
                              batch_size=batch_size, max_iter=10)
    kmeans.fit(X_init)

    # Step 2: Stream remaining data for incremental updates
    print("\nStreaming batches for incremental training...")
    for _, X_batch in reader.stream_batches(batch_size):
        X_batch = normalize(X_batch, axis=1)
        kmeans.partial_fit(X_batch)

    reader.close()
    print("\nTraining complete.")

    # Step 3: Save model
    with open("glove_kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    print("Model saved to glove_kmeans_model.pkl")

if __name__ == "__main__":
    main()
