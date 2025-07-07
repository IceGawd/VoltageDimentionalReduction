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

def test_voltage(voltages, ignore_fraction:float=0.90, thr:float=0.05):
	sorted = np.sort(voltages.flatten())
	low=int(ignore_fraction*sorted.shape[0])
	scaled = (sorted - sorted[low] )/ (sorted[-1] - sorted[low])
	scaled=np.maximum(scaled,0)
	advantage=np.mean(scaled[low:])
	# print(f'advantage={advantage}')
	if advantage>thr:
		scaled_voltages=(voltages - sorted[low])/(sorted[-1] - sorted[low])
		scaled_voltages=np.maximum(scaled_voltages,0)
		print(f'min(scaled_voltages) {min(scaled_voltages.flatten())}')
		return advantage, True, scaled_voltages
	else:
		return advantage, False, None

import faiss

def compute_distances(centroids, voltages):
# Using Faiss compute three types of distances:
# 1. Euclidean distance between points in the point set
# 2. Distance between points in the point set based on voltages
# 3. Distance between points in the point set based on the k-connectivity graph
#    where the distance is defined as the number of hops in the graph 
#    use FAISS to efficiently compute distances
	# 1. Euclidean distance
	# Using the centroids directly
	X = centroids.points
	index = faiss.IndexFlatL2(X.shape[1])  # L2 distance index
	index.add(X.astype(np.float32))  # Add points to the index
	D1, _ = index.search(X.astype(np.float32), X.shape[0])
	np.fill_diagonal(D1, -np.inf)

	# 2. Distance based on voltages
	# using L2 distance

	vectors = voltages.astype(np.float32)
	index = faiss.IndexFlatL2(vectors.shape[1])  # L2 distance index

	index.add(vectors)  # Add points to the index
	D2, _ = index.search(vectors, vectors.shape[0])
	np.fill_diagonal(D2, -np.inf)

	# 3. Distance based on voltages
	# using L1 distance
	#print("vectors.shape:", vectors.shape, flush=True)
	#D3 = np.abs(vectors[:, None, :] - vectors[None, :, :]).sum(axis=2)
	#print("D3 shape:", D3.shape, flush=True)
	#np.fill_diagonal(D3, -np.inf)

	print("Returning shapes:", D1.shape, D2.shape, D3.shape,flush=True)
	return D1, D2, D2


if __name__ == "__main__":
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

	run_kmeans=False

	workspace_file="../../Voltage_Temp/Intermediates/pointset.pkl"
	import dill  # dill is used to save the workspace

	if run_kmeans:
		# generate centroids using streaming k-means
		centroids, counters, majority_labels, _,_=kmeans.Streaming_Kmeans(config.params['file_path'])

		X=np.stack(centroids)
		# Normalize pixel values to [0, 1]
		X = X / 255.0   #for visualization purposes
		y= np.array(majority_labels)

		# define set of points on which we will work
		centroids = setofpoints.SetOfPoints(points=X, weights=counters)


### Store /recover intermediate workspace
		dill.dump_session(workspace_file)
	else:
		dill.load_session(workspace_file)
	
	print("starting building landmarks after kmeans is done")

	# compute the voltage map for each centroid

	all_voltages = voltagemap.VoltageMap()	
	_problem = problem.Problem(centroids,r=0.01)
	_solver=solver.Solver(_problem)

	from time import time
	start_time = time()
	for index in range(len(centroids)):
		_landmark= landmark.Landmark(index, voltage=1.0)
		voltages=_solver.compute_voltages(_landmark)
		all_voltages.add_solution(_landmark, voltages=voltages)
	end_time = time()
	print(f"Computed voltages for {len(centroids)} centroids in {end_time - start_time:.2f} seconds")
	all_voltages.sort_by_norm()  # sort the voltage map by norm in descending order
	print(f"Voltage map has {len(all_voltages)} entries after sorting by norm")

	# Initialize the map
	voltage_map=voltagemap.VoltageMap()
	max_voltage=np.zeros(len(all_voltages))  # to keep track of the maximum voltage for each landmark

	for i, (lm, voltages, norm) in enumerate(all_voltages.entries):
		if i==0:
			# Initialize the voltage map with the first landmark
			voltage_map.add_solution(lm, voltages=voltages)
			continue
		else:
			v1= voltages
			norm1 = norm
			for vm in voltage_map.entries:
				v2= vm[1]
				norm2 = vm[2]
				dp= np.dot(v1, v2)/(norm1 * norm2)  # dot product
				print(f"Landmark {i} - Dot product with existing landmarks: {dp:.4f}")
			break 


	# save the workspace for later use

	workspace_file="../../Voltage_Temp/Intermediates/workspace.pkl"
	dill.dump_session(workspace_file)
	print(f"Workspace saved to {workspace_file}")

	import pickle
	with open(config.params['Voltage_map_output'], 'wb') as f:
		pickle.dump(voltage_map, f)
	print(f"Voltage map saved to {config.params['Voltage_map_output']}")

