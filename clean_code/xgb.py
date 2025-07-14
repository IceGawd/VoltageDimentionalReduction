from pyexpat import model
import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from Utilities.set_params import set_params
from Utilities.reader import Reader, ParseException
from Utilities import config

# ---------- Data Loading Functions ----------
    
def load_voltage_and_centroids(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data['centroids'], data['voltage_map']

# def load_labeled_data(path: str):
#     df = pd.read_csv(path, dtype=str, low_memory=False)
#     df = df[df.iloc[:, 0] != "label"]
#     df = df.astype(np.float32)
#     y = df.iloc[:, 0].astype(int).values
#     X = df.iloc[:, 1:].values
#     return X, y

def stream_and_embed_batches(file_path, centroids, voltage_map):
    reader = Reader(file_path)
    all_features = []
    all_labels = []

    for X_batch, y_batch in reader.stream_batches(config.params['batch_size']):
        y_batch = np.array(y_batch, dtype=int)
        X_embedded = embed_voltage_features(X_batch, centroids, voltage_map)
        all_features.append(X_embedded)
        all_labels.append(y_batch)

    reader.close()

    X_total = np.vstack(all_features)
    y_total = np.concatenate(all_labels)
    return X_total, y_total

# ---------- Voltage Embedding Function ----------

def embed_voltage_features(X_data, centroids, voltage_map, use_rbf=True):
    k = config.params['k']
    sigma = config.params['sigma']
    n_points = X_data.shape[0]
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
    model = XGBClassifier(n_estimators=1000,
                          learning_rate=0.1,
                          objective='multi:softmax', num_class=10,
                           max_depth=5,
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
    from Utilities.timer import Timer
    timer = Timer()
    timer.mark("Loading voltage map and centroids")
    centroids, voltage_map= load_voltage_and_centroids(config.params['Voltage_map_output'])
    X_voltage, y_data = stream_and_embed_batches(config.params['file_path'], centroids, voltage_map)
    timer.mark("Embedded voltage features")
    train_and_evaluate(X_voltage, y_data)
    timer.mark("Training and evaluation completed")

if __name__ == "__main__":
     set_params()
     if config.params['test']:
        config.params['file_path']= '../../Voltage_Data/mnist/mnist.csv'
        config.params['Voltage_map_output'] = "../../Voltage_Temp/Results/voltage_map.npy"
        config.params['split_char']=','
     main()
