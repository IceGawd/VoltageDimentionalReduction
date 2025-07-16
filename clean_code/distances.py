import numpy as np
from Utilities import config
from scipy.sparse import csgraph

def compute_distances(point_set, voltages):
# Using Faiss compute three types of distances:
# 1. Euclidean distance between points in the point set
# 2. Distance between points in the point set based on voltages
# 3. Distance between points in the point set based on the k-connectivity graph
#    where the distance is defined as the number of hops in the graph 
#    use FAISS to efficiently compute distances
	# 1. Euclidean distance
	# Using the point_set directly
	import faiss
	X = point_set.points
	index = faiss.IndexFlatL2(X.shape[1])  # L2 distance index
	index.add(X.astype(np.float32))  # Add points to the index
	D1, _ = index.search(X.astype(np.float32), X.shape[0])
	np.fill_diagonal(D1, -np.inf)

	# 2. Distance based on voltages
	X= voltages
	index = faiss.IndexFlatL2(X.shape[1])  # L2 distance index
	index.add(X.astype(np.float32))  # Add points to the index
	D2, _ = index.search(X.astype(np.float32), X.shape[0])
	np.fill_diagonal(D2, -np.inf)

	# 3. Distance based on k-connectivity graph
	# Using the point_set directly
	# Create a k-nearest neighbors graph
	k = config.params['k']
	from scipy.sparse import lil_matrix
	from sklearn.neighbors import NearestNeighbors

	nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
	_, indices = nbrs.kneighbors(X)
	# Create a sparse adjacency matrix
	n = X.shape[0]
	adjacency_matrix = lil_matrix((n, n), dtype=np.float32)
	for i in range(n):
		for j in indices[i]:
			if i != j:
				adjacency_matrix[i, j] = 1.0
				adjacency_matrix[j, i] = 1.0
	
	D3 = csgraph.shortest_path(adjacency, method = "auto", directed = False, unweighted=True)
	D3[np.isinf(D3)] = np.inf

	return D1, D2, D3




