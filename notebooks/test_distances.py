import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix, csgraph
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import pairwise_distances

np.random.seed(42)
# X = np.random.rand(10, 2)  # 10 points in 2D space

# cluster1 = 0.1 * np.random.rand(5, 2)
# cluster2 = 0.1 * np.random.rand(5, 2) + 0.9
# X = np.vstack([cluster1, cluster2])
# print("Points:\n", X)

cluster_size = 4
spread = 0.2  
centers = np.random.rand(4, 2)
clusters = []
for center in centers:
    cluster_points = spread * np.random.rand(cluster_size, 2) + center
    clusters.append(cluster_points)
X = np.vstack(clusters)
print("Points shape:", X.shape)

D1 = pairwise_distances(X, metric='euclidean')  

k = 3
nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
_, indices = nbrs.kneighbors(X)

n = X.shape[0]
adjacency_matrix = lil_matrix((n, n), dtype=np.float32)
for i in range(n):
    for j in indices[i]:
        if i != j:
            adjacency_matrix[i, j] = 1.0
            adjacency_matrix[j, i] = 1.0

n_components, labels = connected_components(adjacency_matrix, directed=False)
print(f"Number of connected components: {n_components}")
print(f"Component labels for each node: {labels}")

D3 = csgraph.shortest_path(adjacency_matrix, method="auto", directed=False, unweighted=True)
D3[np.isinf(D3)] = np.inf

print("Euclidean distance (D1):\n", np.round(D1, 2))
print("\nGraph-geodesic distance (D3):\n", D3)

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], color='black', s=50, zorder=3)

for idx, (x, y) in enumerate(X):
    plt.text(x + 0.01, y + 0.01, str(idx), color='red', fontsize=12)
for i in range(n):
    for j in indices[i]:
        if i < j: 
            plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'b--', alpha=0.5)

plt.title("k-NN Graph (k=3) with Point Labels")
plt.grid(True)
plt.tight_layout()
plt.savefig("/mntdata/main/dev/voltage/VoltageDimentionalReduction/notebooks/knn_graph.png", dpi=300)
plt.close()
