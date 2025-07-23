from pyexpat import model
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
import io
from sklearn.preprocessing import LabelEncoder
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
    return data['centroids'], data['voltage_map']

def load_labeled_data(path: str, n_rows: int = None):
    cleaned_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n_rows is not None and i >= n_rows:
                break
            # Remove inline comments
            line = line.split('#')[0].strip()
            if not line:
                continue  # skip empty lines
            cleaned_lines.append(line)
    # Convert to a string buffer and use pandas
    cleaned_text = "\n".join(cleaned_lines)
    buffer = io.StringIO(cleaned_text)
    df = pd.read_csv(buffer, sep=config.params['split_char'],dtype=str, low_memory=False, nrows=n_rows)
    df = df[df.iloc[:, 0] != "label"]
    try:
        y = df.iloc[:, 0].astype(int).values
    except ValueError:
        y = df.iloc[:, 0].values
        le = LabelEncoder()
        y = le.fit_transform(y)
        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print("Label mapping:", label_mapping)
    X = df.iloc[:, 1:].astype(np.float32).values
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
    for i, entry in enumerate(voltage_map.entries):
        v_vector = entry['voltages']
        for j in range(n_points):
            neighbor_ids = indices[j]
            neighbor_voltages = v_vector[neighbor_ids]
            weights = weights_all[j]
            features[j, i] = np.dot(weights, neighbor_voltages)

    return features

# ---------- Modeling Function ----------

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=2000,
                          learning_rate=0.1,
                          objective='multi:softmax', num_class=10,
                           max_depth=2,
                           eval_metric='merror',  # specify metric
                           random_state=42)
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train,eval_set=eval_set)

    results = model.evals_result()
    from Utilities.xgb_util import plot_train_test_errors
    plot_train_test_errors(results)
    
    train_score = model.score(X_train, y_train)
    print("Train accuracy:", train_score)
    # Evaluate on test set
    test_score = model.score(X_test, y_test)
    print("Test accuracy:", test_score)
    return model

# ---------- Main Block ----------

def main():
    voltage_map_path = config.params['save_data']
    data_path = config.params['file_path']
    n_rows = config.params['n_rows']
    from Utilities.timer import Timer
    timer = Timer()
    timer.mark("Loading voltage map and centroids")
    centroids, voltage_map= load_voltage_and_centroids(voltage_map_path)
    X_data, y_data = load_labeled_data(data_path,n_rows)

    X_voltage = embed_voltage_features(X_data, centroids, voltage_map,sigma=config.params['sigma'])
    timer.mark("Embedded voltage features")
    train_and_evaluate(X_voltage, y_data)
    timer.mark("Training and evaluation completed")
if __name__ == "__main__":
    from Utilities.set_params import set_params
    set_params()  # Load configuration parameters
    main()
