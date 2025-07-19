import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier 
from Utilities.xgb_util import plot_train_test_errors
from Utilities import config
from Utilities.reader import Reader, ParseException
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
    return data['centroids'], data['voltage_map'], data['k']  

def load_labeled_data(path: str):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = df[df.iloc[:, 0] != "label"]
    df = df.astype(np.float32)
    y = df.iloc[:, 0].astype(int).values
    X = df.iloc[:, 1:].values
    return X, y

all_pos_tags = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
]
label_encoder = LabelEncoder()
label_encoder.fit(all_pos_tags)

# ---------- Voltage Embedding Function ----------

def embed_voltage_features(X_data, centroids, voltage_map, use_rbf=True, sigma=None):
    k = config.params['k']
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
    for i, entry in enumerate(voltage_map.entries):
        v_vector = entry['voltages']
        for j in range(n_points):
            neighbor_ids = indices[j]
            neighbor_voltages = v_vector[neighbor_ids]
            weights = weights_all[j]
            features[j, i] = np.dot(weights, neighbor_voltages)

    return features

#-----------Streaming Training ----------

def stream_voltage_xgb_training(centroids, voltage_map):
    reader = Reader(config.params['file_path'])
    X_test, y_test = next(reader.stream_batches(config.params['batch_size'])) #using first batch as test set
    X_test = embed_voltage_features(X_test, centroids, voltage_map)
    if np.all([label.isdigit() for label in y_test]):
        y_test = y_test.astype(np.int32)
    else:
        y_test = label_encoder.transform(y_test)
    params = {
        'objective': 'multi:softmax',
        'num_class': 17,
        # 'num_class': 10,
        'eval_metric': 'merror',
        'learning_rate': 0.1,
        'max_depth': 2,
        'random_state': 42,
        'tree_method': 'hist',
        'nthread': 2
        }
    # num_boost_round = 200
    iteration = 0
    trees_per_batch = 1  # number of trees to add per batch
    model = None
    train_errors = []
    test_errors = []
    for batch_idx, (X_batch, y_batch) in enumerate(reader.stream_batches(config.params['batch_size'])):
        if X_batch is None or y_batch is None or len(X_batch) == 0:
            break
        X_embed = embed_voltage_features(X_batch, centroids, voltage_map)
        if np.all([label.isdigit() for label in y_batch]):
            y_batch = y_batch.astype(np.int32)
        else:
            y_batch = label_encoder.transform(y_batch)
        dtrain = xgb.DMatrix(X_embed, label=y_batch)

        # # Train batch
        # evals_result = {}
        # model = xgb.train(
        #     params = params,
        #     dtrain = dtrain,
        #     num_boost_round=num_boost_round,
        #     xgb_model=model,  # continue from previous
        #     evals=[(dtrain, "train")], # logs error on this batch
        #     evals_result = evals_result
        # )
        # print(f" Batch {batch_idx+1} trained.")
        
        # Initialize Booster once
        if model is None:
            model = xgb.Booster(params=params, cache=[dtrain])

        # Update booster with 1 or more trees
        for _ in range(trees_per_batch):
            model.update(dtrain, iteration)
            iteration += 1

        y_pred_train = model.predict(dtrain)
        train_acc = np.mean(y_pred_train == y_batch)
        train_errors.append(1 - train_acc)
        print(f"Train accuracy on batch {batch_idx + 1}: {train_acc:.4f}")
        
        dtest = xgb.DMatrix(X_test)
        y_pred_test = model.predict(dtest)
        test_acc = np.mean(y_pred_test == y_test)
        test_errors.append(1 - test_acc)

    reader.close()
    model.save_model("../../Voltage_Temp/Results/mnist_voltage_streamed_model.json")
    print(" Model saved to mnist_voltage_streamed_model.json")
    # results = model.evals_result()
    # plot_train_test_errors(results)    
    plot_train_test_errors(train_errors, test_errors)
    # dtest = xgb.DMatrix(X_test)
    # # Predict using the Booster
    # y_pred = model.predict(dtest)
    # # Compute test accuracy manually
    # test_accuracy = np.mean(y_pred == y_test)
    # print("Test accuracy:", test_accuracy)
    return model

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
    if config.params['test']:
     
        # voltage_map_path = "../../Voltage_Temp/Results/voltage_map.npy"
        # data_path = "../../Voltage_Data/mnist/mnist.csv"
        # config.params['file_path'] = '../Voltage_Data/mnist/mnist.csv'
        # config.params['split_char']=','

        voltage_map_path = '../../Voltage_Temp/Results/glove/voltage_map.npy'
        data_path =  '../../Voltage_Data/glove/glove_with_pos.txt'
        config.params['file_path']= '../../Voltage_Data/glove/glove_with_pos.txt'

    else:
        # call select_landmarks here
        voltage_map_path = config.params['voltage_map']
        data_path = config.params['data']

    from Utilities.timer import Timer
    timer = Timer()
    timer.mark("Loading voltage map and centroids")
    centroids, voltage_map, k = load_voltage_and_centroids(voltage_map_path)
    # config.params['k'] = k 
    # X_data, y_data = load_labeled_data(data_path)

    # X_voltage = embed_voltage_features(X_data, centroids, voltage_map,sigma=config.params['sigma'])
    # timer.mark("Embedded voltage features")
    # train_and_evaluate(X_voltage, y_data)
    # timer.mark("Training and evaluation completed")
    timer.mark("Embedded voltage features")
    stream_voltage_xgb_training(centroids, voltage_map)
    timer.mark("Training and evaluation completed")

if __name__ == "__main__":
    from Utilities.set_params import set_params
    set_params()  # Load configuration parameters
    main()
