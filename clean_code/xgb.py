import numpy as np
import pandas as pd
import pickle
import dill
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import argparse
import config


# ---------- Data Loading Functions ----------

def load_voltage_map(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_point_set(path: str):
    with open(path, "rb") as f:
        return dill.load(f)


def load_labeled_data(path: str):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = df[df.iloc[:, 0] != "label"]
    df = df.astype(np.float32)
    y = df.iloc[:, 0].astype(int).values
    X = df.iloc[:, 1:].values
    return X, y


# ---------- Voltage Embedding Function ----------

def embed_voltage_features(X_data, centroids, voltage_map,sigma):
    k = config.params['k']
    sigma = sigma
    use_rbf = True
    n_points = X_data.shape[0]
    n_landmarks = len(voltage_map.voltage_maps)
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

    for l_idx in range(n_landmarks):
        v_vector = voltage_map.get_solution(l_idx)
        for i in range(n_points):
            neighbor_ids = indices[i]
            neighbor_voltages = v_vector[neighbor_ids]
            weights = weights_all[i]
            features[i, l_idx] = np.dot(weights, neighbor_voltages)

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
    global point_set
    if args.test_only:
        print("Running MNIST test-only mode...")
        voltage_map = load_voltage_map("../../Voltage_Temp/Results/voltage_map.npy")
        #point_set = load_point_set("../../Voltage_Temp/Intermediates/pointset.pkl")
        dill.load_session("../../Voltage_Temp/Intermediates/workspace.pkl")
        X_data, y_data = load_labeled_data("../../Voltage_Data/mnist/mnist.csv")
        sigma = None
        X_voltage = embed_voltage_features(X_data, point_set.points, voltage_map,sigma)
        train_and_evaluate(X_voltage, y_data)
        return

    voltage_map = load_voltage_map(args.voltage_map)
    #point_set = load_point_set(args.pointset)
    dill.load_session(args.pointset)

    X_data, y_data = load_labeled_data(args.data)
    centroid_vectors = point_set.points
    sigma = args.sigma
    X_voltage = embed_voltage_features(X_data, centroid_vectors, voltage_map, sigma)

    _ = train_and_evaluate(X_voltage, y_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voltage-based XGBoost classifier")
    parser.add_argument("--data", type=str, help="Path to labeled CSV file")
    parser.add_argument("--voltage_map", type=str, help="Path to voltage_map.npy file")
    parser.add_argument("--pointset", type=str, help="Path to pointset.pkl or workspace.pkl file")
    parser.add_argument("-T", "--test_only", action="store_true", help="Run in test-only mode")
    parser.add_argument("--sigma", type=float, help="RBF sigma value")

    args = parser.parse_args()

    if not args.test_only:
        if not args.data or not args.voltage_map or not args.pointset:
            parser.error("In normal mode, --data, --voltage_map, and --pointset are required.")

    main(args)
