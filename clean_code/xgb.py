import numpy as np
import pandas as pd
import pickle
import dill
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import argparse
from Utilities import config
import os
import subprocess
import sys

# ---------- Data Loading Functions ----------

# def load_voltage_map(path: str):
#     with open(path, "rb") as f:
#         return pickle.load(f)


# def load_point_set(path: str):
#     with open(path, "rb") as f:
#         return dill.load(f)
    
def load_voltage_and_centroids(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data['centroids'], data['voltage_map'], data.get('k', 5)  # default to 5

def load_labeled_data(path: str):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = df[df.iloc[:, 0] != "label"]
    df = df.astype(np.float32)
    y = df.iloc[:, 0].astype(int).values
    X = df.iloc[:, 1:].values
    return X, y

# ---------- Voltage Embedding Function ----------

def embed_voltage_features(X_data, centroids, voltage_map, use_rbf=True, sigma=None):
    k = config.params['k']
    n_points = X_data.shape[0]
    print(type(voltage_map))
    n_landmarks = len(voltage_map)
    print("Number of landmarks:", n_landmarks)
    features = np.zeros((n_points, n_landmarks))

    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(centroids)
    distances, indices = knn.kneighbors(X_data)

    if use_rbf:
        if sigma is None:
            sigma = np.median(distances)
        weights_all = np.exp(-distances**2 / (2 * sigma**2))
    else:
        weights_all = 1 / (distances + 1e-8)

    weights_all /= np.sum(weights_all, axis=1, keepdims=True)
    # Don't read all of the data at once, instead read it in chunks using Reader
    for i, (landmark_obj, v_vector, _) in enumerate(voltage_map.entries):
        for j in range(n_points):
            neighbor_ids = indices[j]
            neighbor_voltages = v_vector[neighbor_ids]
            weights = weights_all[j]
            features[j, i] = np.dot(weights, neighbor_voltages)

    return features

# ---------- Modeling Function ----------

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print("Test accuracy:", test_score)
    return model

# ---------- Main Block ----------

def main(args):
    if args.test_only:
        print("Running MNIST test-only mode...")

        # 1. Delete voltage_map result file
        voltage_map_output = '../../Voltage_Temp/Results/voltage_map.npy'
        if os.path.exists(voltage_map_output):
            try:
                os.remove(voltage_map_output)
                print(f"Deleted voltage map output: {voltage_map_output}")
            except Exception as e:
                print(f"Failed to delete {voltage_map_output}: {e}")
        else:
            print(f"No existing voltage map output found at {voltage_map_output}")

        # 2. Run main.py
        print("Running main.py to regenerate workspace and voltage map...")
        subprocess.run([sys.executable, "main.py", "--test"], check=True) #sys.executable uses the venv

        # 3. Set file paths
        voltage_map_path = "../../Voltage_Temp/Results/voltage_map.npy"
        data_path = "../../Voltage_Data/mnist/mnist.csv"
    else:
        # call select_landmarks here
        voltage_map_path = args.voltage_map
        data_path = args.data

    centroids, voltage_map, k = load_voltage_and_centroids(voltage_map_path)
    config.params['k'] = k 
    X_data, y_data = load_labeled_data(data_path)

    X_voltage = embed_voltage_features(X_data, centroids, voltage_map,sigma=args.sigma)
    train_and_evaluate(X_voltage, y_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voltage-based XGBoost classifier")
    parser.add_argument("--data", type=str, help="Path to labeled CSV file")
    parser.add_argument("--voltage_map", type=str, help="Path to npy file containing {'centroids', 'voltage_map', 'k'}")
    parser.add_argument("-T", "--test_only", action="store_true", help="Run in test-only mode")
    parser.add_argument("--sigma", type=float, default=None, help="Sigma value for RBF weighting (default: auto)")

    args = parser.parse_args()
    main(args)
