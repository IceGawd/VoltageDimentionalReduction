import numpy as np
from sklearn.cluster import MiniBatchKMeans

class Reader:
    def __init__(self, path):
        self.file = open(path, 'r')
        self.counter = 0

    def stream_batches(self, batch_size):
        while True:
            L = []
            words = []
            for _ in range(batch_size):
                line = self.file.readline()
                if line == '':
                    break
                T = line.strip().split()
                words.append(T[0])
                vec = np.array([float(x) for x in T[1:]])
                L.append(vec)
                self.counter += 1
                print(self.counter, end='\r')
            if not L:
                break
            Vecs=np.stack(L).astype('float32')
            Vecs=Vecs/np.linalg.norm(Vecs,axis=1,keepdims=True)
            yield np.array(words), Vecs

    def close(self):
        self.file.close()

# Parameters
import kagglehub
path = kagglehub.dataset_download("thanakomsn/glove6b300dtxt")
glove_path = path + "/glove.6B.300d.txt" 
#glove_path = "shuffled_output.txt"
path = glove_path
k = 1000           # Number of clusters
init_size = 10000  # Number of points to initialize k-means
batch_size = 1000  # Batch size for streaming

# Initialize the reader
reader = Reader(path)

# Read initial batch to initialize centroids
_, init_batch = next(reader.stream_batches(init_size))
print(f"\nInitializing KMeans on {init_batch.shape[0]} samples")

# Initialize MiniBatchKMeans with pre-fit centroids
kmeans = MiniBatchKMeans(n_clusters=k, init='k-means++', batch_size=batch_size, max_iter=10)
kmeans.fit(init_batch)

# Stream the rest of the file
for _, batch in reader.stream_batches(batch_size):
    kmeans.partial_fit(batch)

reader.close()
print("\nFinal cluster centers shape:", kmeans.cluster_centers_.shape)

import pickle
with open('kmeans.pkl','wb') as pkl:
    pickle.dump(kmeans.cluster_centers_,pkl)
