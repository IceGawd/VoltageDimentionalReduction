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
from Utilities import config
import faiss

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


def compute_distances(centroids, voltages):
	"""
	Compute three types of pairwise distances between points using Faiss.
	This function calculates:
		1. Euclidean distances between points in the centroid set.
		2. Euclidean (L2) distances between points based on their voltage representations.
		3. (Commented out) L1 distances between points based on their voltage representations.
	The function uses Faiss for efficient computation of L2 distances.
	Args:
		centroids: A pointset object with `.points` attribute, representing the coordinates of the centroids as a list of NumPy arrays of shape (n_features).
		voltages:  a voltageMap.VoltageMap object containing voltage vectors for each landmark.
	Returns:
		tuple:
			D1 (np.ndarray): Pairwise Euclidean distances between centroids (with diagonal set to -inf).
			D2 (np.ndarray): Pairwise Euclidean distances between voltage vectors (with diagonal set to -inf).
	Note:
		- The third distance matrix (L1 distance) is currently commented out and not returned.
		- The diagonal of each distance matrix is set to -inf to ignore self-distances.
	"""

	if not isinstance(centroids, setofpoints.SetOfPoints):
		raise TypeError("centroids must be an instance of setofpoints.SetOfPoints")
	if not isinstance(voltages, voltagemap.VoltageMap):
		raise TypeError("voltages must be an instance of voltagemap.VoltageMap")
	# 1. Euclidean distance
	# Using the centroids directly
	print("Computing distances...", flush=True)
	X = np.stack(centroids.points)  # Expecting centroids to have a .points attribute
	print(f"type(X): {type(X)}", flush=True)
	print("X.shape:", X.shape, flush=True)
	index = faiss.IndexFlatL2(X.shape[1])  # L2 distance index
	index.add(X.astype(np.float32))  # Add points to the index
	D1, _ = index.search(X.astype(np.float32), X.shape[0])
	np.fill_diagonal(D1, -np.inf)

	# 2. Distance based on voltages
	# using L2 distance

	vectors = voltages.all_solutions().astype(np.float32)  # Convert to float32 for Faiss compatibility
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

	#print("Returning shapes:", D1.shape, D2.shape, D3.shape,flush=True)
	Ds=[D1, D2]
	print(len(Ds), flush=True)
	return Ds


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
	config.params['saved_data']= '../../Voltage_Temp/Results/saved_data.pkl'
	config.params['k']=10

	run_kmeans=False

	workspace_file="../../Voltage_Temp/Intermediates/pointset.pkl"
	import dill  # dill is used to save the workspace

	if run_kmeans:
		# generate centroids using streaming k-means
		points, counters, majority_labels, _,_=kmeans.Streaming_Kmeans(config.params['file_path'])

		X=np.stack(points)
		y= np.array(majority_labels)

		# define set of points on which we will work
		centroids = setofpoints.SetOfPoints(points=points, weights=counters)

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
		all_voltages.sort_by_advantage()  # sort the voltage map by norm in descending order
		print(f"Voltage map has {len(all_voltages)} entries after sorting by norm")



### Store /recover intermediate workspace
		dill.dump_session(workspace_file)
	else:
		dill.load_session(workspace_file)
	
	print("starting building landmarks after kmeans is done")

	# Initialize the map
	voltage_map=voltagemap.VoltageMap()
	lm, voltages, _ = all_voltages.entries[0]  # get the first landmark and its voltages
	voltage_map.add_solution(lm, voltages=voltages)
	max_voltage=np.zeros(len(all_voltages))  # to keep track of the maximum voltage for each landmark

# repeatedly iteration all_voltages.entries and add the landmark with the largest distance to the selected landmarks to the voltage map	
	for iteration in range(100):
		# Find the landmark in all_voltages.entries that is farthest from the current voltage_map entries
		max_min_dist = 2.0
		best_idx = None
		best_norm = 2.0
		for idx, (lm, voltages, norm) in enumerate(all_voltages.entries):
			# Skip if already in voltage_map
			if any(np.array_equal(lm.index, vmap_lm.index) for vmap_lm, _, _ in voltage_map.entries):
				continue	
			# Compute minimum distance to any entry in voltage_map
			min_dist = np.min([np.linalg.norm(voltages - vm[1]) for vm in voltage_map.entries])
			if min_dist > max_min_dist and norm> best_norm:
				max_min_dist = min_dist
				best_idx = idx
				best_norm = norm
		print(f"Iteration {iteration}: Best landmark index {best_idx} norm={best_norm:.4f} with min distance {max_min_dist:.4f}")
		if best_idx is not None:
			lm, voltages, norm = all_voltages.entries[best_idx]
			voltage_map.add_solution(lm, voltages=voltages)
		else:
			break

	print("About to call compute_distances")
	#import pdb
	#pdb.set_trace()
	Ds = compute_distances(centroids, voltage_map)
	print(f"len(Ds) = {len(Ds)}")
	print(f"Computed distances: D1 shape {Ds[0].shape}, D2 shape {Ds[1].shape}")
	Deuc =Ds[0]
	Dvolt = Ds[1]
	
	# save the workspace for later use

	workspace_file="config.params['save_data']"
	dill.dump_session(workspace_file)
	print(f"Workspace saved to {workspace_file}")

	import pickle
	with open(config.params['saved_data'], 'wb') as f:
		pickle.dump(voltage_map, f)
	print(f"Voltage map saved to {config.params['saved_data']}")

