import numpy as np
import pickle
import dill
from sklearn.neighbors import NearestNeighbors
import os
import pandas as pd


# --- Load voltage_map and centroids (point_set) ---

# Load VoltageMap
with open("../../Voltage_Temp/Results/voltage_map.npy", "rb") as f:
    voltage_map = pickle.load(f)

# Load point_set (centroid locations)
with open("../../Voltage_Temp/Intermediates/pointset.pkl", "rb") as f:
    dill.load_session(f)  # this loads `point_set` into the environment

# Extract centroid vectors
centroid_vectors = point_set.points  # shape: (n_centroids, d)
n_landmarks = len(voltage_map.voltage_maps)

# --- Load full dataset ---
# Load everything as strings first to prevent errors
df = pd.read_csv("../../Voltage_Data/mnist/mnist.csv", dtype=str, low_memory=False)

# Remove any rows where the first column is 'label' (i.e., header rows)
df = df[df.iloc[:, 0] != "label"]

# Convert all remaining values to float
df = df.astype(np.float32)

# Split into labels and features
y_data = df.iloc[:, 0].astype(int).values
X_data = df.iloc[:, 1:].values

# --- Voltage-based feature embedding using weighted average ---

def embed_voltage_features(X_data, centroids, voltage_map, k=5, use_rbf=True, sigma=None):
    """
    For each point in X_data, find its k nearest centroids and compute
    voltage-based features using weighted avg of those centroids.
    """
    n_points = X_data.shape[0]
    n_landmarks = len(voltage_map.voltage_maps)
    print(n_landmarks)
    features = np.zeros((n_points, n_landmarks))

    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(centroids)

    distances, indices = knn.kneighbors(X_data)

    # Compute weights
    if use_rbf:
        if sigma is None:
            sigma = np.median(distances)
        weights_all = np.exp(-distances**2 / (2 * sigma**2))
    else:
        weights_all = 1 / (distances + 1e-8)  # avoid division by zero

    weights_all /= np.sum(weights_all, axis=1, keepdims=True)  # normalize

    for l_idx in range(n_landmarks):
        v_vector = voltage_map.get_solution(l_idx)  # shape: (n_centroids,)
        for i in range(n_points):
            neighbor_ids = indices[i]              # shape: (k,)
            neighbor_voltages = v_vector[neighbor_ids]
            weights = weights_all[i]               # shape: (k,)
            features[i, l_idx] = np.dot(weights, neighbor_voltages)

    return features

# --- Run it ---

X_voltage = embed_voltage_features(X_data, centroid_vectors, voltage_map, k=5)
print("Voltage-based feature matrix shape:", X_voltage.shape)

# Train/test split and model
# from sklearn.model_selection import train_test_split
# from xgboost import XGBClassifier

# X_train, X_test, y_train, y_test = train_test_split(X_voltage, y_data, test_size=0.2, random_state=42)

# model = XGBClassifier()
# model.fit(X_train, y_train)
# print("Test accuracy:", model.score(X_test, y_test))
