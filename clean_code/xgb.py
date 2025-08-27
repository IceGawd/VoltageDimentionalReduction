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
from Utilities import voltage_embed
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
	 
		voltage_map_path = "../../Voltage_Temp/Results/saved_data.pkl"
		data_path = "../../Voltage_Data/mnist/mnist.csv"
	else:
		# call select_landmarks here
		voltage_map_path = config.params['voltage_map']
		data_path = config.params['data']

	from Utilities.timer import Timer
	timer = Timer()
	timer.mark("Loading voltage map and centroids")
	centroids, voltage_map, k = load_voltage_and_centroids(voltage_map_path)
	config.params['k'] = k 
	X_data, y_data = load_labeled_data(data_path)

	X_voltage = voltage_embed.embed_voltage_features(X_data, centroids, voltage_map,sigma=config.params['sigma'])
	timer.mark("Embedded voltage features")
	train_and_evaluate(X_voltage, y_data)
	timer.mark("Training and evaluation completed")
if __name__ == "__main__":
	from Utilities.set_params import set_params
	set_params()  # Load configuration parameters
	main()
