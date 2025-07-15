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
from Utilities.distances import compute_distances
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

	workspace_file="../../Voltage_Temp/Intermediates/workspace.pkl"
	dill.dump_session(workspace_file)
	print(f"Workspace saved to {workspace_file}")

	import pickle
	with open(config.params['Voltage_map_output'], 'wb') as f:
		pickle.dump(voltage_map, f)
	print(f"Voltage map saved to {config.params['Voltage_map_output']}")

