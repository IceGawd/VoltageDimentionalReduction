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

def test_voltage(voltages, ignore_fraction:float=0.95, thr:float=0.05):
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

def compute_distances(point_set, voltages):
# Using Faiss compute three types of distances:
# 1. Euclidean distance between points in the point set
# 2. Distance between points in the point set based on voltages
# 3. Distance between points in the point set based on the k-connectivity graph
#    where the distance is defined as the number of hops in the graph 
#    use FAISS to efficiently compute distances
	# 1. Euclidean distance
	# Using the point_set directly
	import faiss
	X = point_set.points
	index = faiss.IndexFlatL2(X.shape[1])  # L2 distance index
	index.add(X.astype(np.float32))  # Add points to the index
	D1, _ = index.search(X.astype(np.float32), X.shape[0])
	np.fill_diagonal(D1, -np.inf)

	# 2. Distance based on voltages
	X= voltages
	index = faiss.IndexFlatL2(X.shape[1])  # L2 distance index
	index.add(X.astype(np.float32))  # Add points to the index
	D2, _ = index.search(X.astype(np.float32), X.shape[0])
	np.fill_diagonal(D2, -np.inf)

	# 3. Distance based on k-connectivity graph
	# Using the point_set directly
	# Create a k-nearest neighbors graph
	k = config.params['k']
	from scipy.sparse import lil_matrix
	from sklearn.neighbors import NearestNeighbors

	nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
	_, indices = nbrs.kneighbors(X)
	# Create a sparse adjacency matrix
	n = X.shape[0]
	adjacency_matrix = lil_matrix((n, n), dtype=np.float32)
	for i in range(n):
		for j in indices[i]:
			if i != j:
				adjacency_matrix[i, j] = 1.0





if __name__=="__main__":
	#config.params['file_path']= '../data/glove/shuffled_output.txt'
	#config.params['split_char']= ' '
	#config.params['normalize_vecs']= True

	config.params['file_path']= '../data/mnist/mnist.csv'
	config.params['split_char']= ','
	config.params['normalize_vecs']= False

	config.params['max_centroids']= 1000
	config.params['init_size']= 5000
	config.params['batch_size']= 1000
	config.params['output']= 'streaming_centroids.npy'
	config.params['k']=10

	# generate centroids using streaming k-means
#	centroids, counters, majority_labels, _,_=kmeans.Streaming_Kmeans(config.params['file_path'])

#	X=np.stack(centroids)
#	print('X.shape=',X.shape)
#	# Normalize pixel values to [0, 1]
#	X = X / 255.0   #for visualization purposes
#	y= np.array(majority_labels)

#	# define set of points on which we will work
#	point_set = setofpoints.SetOfPoints(points=X, weights=counters)

	import pickle
#	with  open('../../Voltage_Temp/Intermediates/pointset.pkl','wb') as pkl:
#		pickle.dump(point_set,pkl)
	with  open('../../Voltage_Temp/Intermediates/pointset.pkl','rb') as pkl:
		point_set=pickle.load(pkl)

	print('started')
	#choose landmarks one at a time, starting with a random centroid and then choosing a centroid where all of the voltages so far are low.

	import random

	j=0
	# Initialize the map
	landmarks=[]
	voltage_map = voltagemap.VoltageMap()
	problem = problem.Problem(point_set)
	solver=solver.Solver(problem)
	max_voltage=np.zeros(point_set.__len__())

	Voltage_thr=0.1     # maximal voltage for adding a landmark
	coverage_threshold=0.95  # minimal coverage to terminate the program

	while True:
		index=random.randint(0, len(point_set)-1)  # choose a random point
		if(max_voltage[index]>Voltage_thr):  # check whether it already has significant voltage
			continue
		# choose next landmark to add
		candidate_landmark=landmark.Landmark(index,voltage=1.0)
		voltages=solver.compute_voltages(candidate_landmark)
		advantage,test_passed, scaled_voltages = test_voltage(voltages) 
		if not test_passed:
			continue

		j+=1
		landmarks.append(candidate_landmark)

		# collect, for each point, the voltages calculated so far
		voltage_map.add_solution(landmark_index=candidate_landmark.index, voltages=scaled_voltages)

		voltages_so_far = np.stack(voltage_map.voltage_maps)
		max_voltage=np.max(voltages_so_far,axis=0)
		aver_max=np.mean(max_voltage.flatten())
		min_max=np.min(max_voltage.flatten())
		non_zero_fraction =  np.mean(max_voltage.flatten()>0)
		print(f"iter={j}, advantage={advantage},\
		 aver_max={aver_max}, min_max={min_max},\
			non_zero_fraction={non_zero_fraction}")
		if non_zero_fraction>=coverage_threshold: 
			print(f'non_zero_fraction >= {coverage_threshold}, Stopping')
			break

	#call a function for computing distances between all poirs of points in pointset
	distances = compute_distances(point_set,voltages_so_far)



	# save the workspace for later use
	import dill  # or use 'pickle' for simpler objects
	with open("../../Voltage_Temp/Intermediates/workspace.pkl", "wb") as f:
		dill.dump_session(f)
	print("main complete. Workspace saved to '../../Voltage_Temp/Intermediates/workspace.pkl'.")