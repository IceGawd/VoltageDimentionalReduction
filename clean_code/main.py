import time
import numpy as np
from typing import Union, Optional, List, Any, Tuple, Callable, Dict
from itertools import product
import pandas
import matplotlib.pyplot as plt

from scipy.sparse.linalg import cg
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.datasets import fetch_openml

import importlib

import landmark
import voltagemap
import problem
import solver
import visualization
import setofpoints
import kmeans
import config
import faiss


def compute_voltages(centroids):
	""" compute the voltage map for each centroid """
	all_voltages = voltagemap.VoltageMap()	
	_problem = problem.Problem(centroids,r=0.01)
	_solver=solver.Solver(_problem)
	for index in range(len(centroids)):
		_landmark= landmark.Landmark(index, voltage=1.0)
		voltages=_solver.compute_voltages(_landmark)
		all_voltages.add_solution(_landmark, voltages=voltages)
	return all_voltages

def main():
	# generate centroids using streaming k-means
	points, counters, majority_labels, _,_=kmeans.Streaming_Kmeans(config.params['file_path'])

	X=np.stack(points)
	y= np.array(majority_labels)

	# define set of set of centroids
	centroids = setofpoints.SetOfPoints(points=points, weights=counters)

	# compute voltages for each centroid
	from time import time
	start_time = time()
	all_voltages = compute_voltages(centroids)
	print(f"all_voltages.all_solutions().shape: {all_voltages.all_solutions().shape}")
	end_time = time()
	print(f"Computed voltages for {len(centroids)} centroids in {end_time - start_time:.2f} seconds")

	start_time = time()
	from select_landmarks import select_landmarks
	voltage_map=select_landmarks(all_voltages)
	end_time= time()
	print(f"Selected landmarks in {end_time - start_time:.2f} seconds")

	# print("About to call compute_distances")
	# Ds = compute_distances(centroids, voltage_map)
	# save the workspace for use in
	
	workspace_file="../../Voltage_Temp/Intermediates/workspace.pkl"
	import dill
	with open(workspace_file, "wb") as f:
		dill.dump({
			"majority_labels": majority_labels,
			"centroids": centroids,
			"all_voltages": all_voltages,
			"voltage_map": voltage_map
		}, f)
	print(f"Saved main() variables to {workspace_file}")


	import pickle
	with open(config.params['Voltage_map_output'], 'wb') as f:
		pickle.dump(voltage_map, f)
	print(f"Voltage map saved to {config.params['Voltage_map_output']}")

if __name__ == "__main__":
	from set_params import set_params
	set_params()
	if config.params['test']:
		# Load configuration parameters
		#config.params['file_path']= '../data/glove/shuffled_output.txt'
		#config.params['split_char']= ' '
		#config.params['normalize_vecs']= True

		config.params['file_path']= '../../Voltage_Data/mnist/mnist.csv'
		config.params['split_char']= ','
		config.params['normalize_vecs']= False

		config.params['max_centroids']= 1000
		config.params['init_size']= 5000
		config.params['batch_size']= 1000
		config.params['kmeans_output']= '../../Voltage_Temp/Results/streaming_centroids.npy'
		config.params['Voltage_map_output']= '../../Voltage_Temp/Results/voltage_map.npy'
		config.params['k']=10

		main()
