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


config.params['file_path']= '../data/mnist/mnist.csv'
config.params['split_char']= ','
config.params['normalize_vecs']= False
config.params['max_centroids']= 1000
config.params['init_size']= 10000
config.params['batch_size']= 10000
config.params['output']= 'streaming_centroids.npy'

# generate centroids using streaming k-means
centroids, counters, majority_labels, _,_=kmeans.Streaming_Kmeans(config.params['file_path'])

X=np.stack(centroids)
print('X.shape=',X.shape)
# Normalize pixel values to [0, 1]
X = X / 255.0   #for visualization purposes
y= np.array(majority_labels)

# define set of points on which we will work
point_set = setofpoints.SetOfPoints(points=X, weights=counters)

#choose landmarks one at a time, starting with a random centroid and then choosing a centroid where all of the voltages so far are low.

import random
landmarks = []
this_landmark=landmark.Landmark(random.randint(0, centroids.shape[0]),1.0)
# Initialize the map
voltage_map = voltagemap.VoltageMap()

problem = problem.Problem(point_set)

while True:
	landmarks.append(this_landmark)
	# Find best r and Add the landmark to the voltage map
	best_r,voltages=problem.optimize([this_landmark], k=2)
	voltage_map.add_solution(landmark_index=this_landmark.index, voltages=voltages)

	# choose next landmark to add
	# collect, for each point, the voltages calculated so far
	voltages_so_far = np.stack(voltage_map.voltage_maps)
	print('voltages_so_far.shape=', voltages_so_far.shape)
	# find the point with the lowest voltage so far
	break

visualization.Visualization.plot_mds_digits([2, 3, 4, 5, 7, 8, 9], voltage_map, point_set, y[:1000], alpha_actual=0.5, out_file="../inputoutput/matplotfigures/mnist_mds.png")