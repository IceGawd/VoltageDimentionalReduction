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


print("Streaming K-means on MNIST dataset")

config.params['file_path']= '../data/mnist.csv'
config.params['split_char']= ','
config.params['normalize_vecs']= False
config.params['max_centroids']= 1000
config.params['init_size']= 1000
config.params['batch_size']= 10000
config.params['output']= 'streaming_centroids.npy'

centroids,counters,inital_mean_d2,mean_d2=kmeans.Streaming_Kmeans(config.params['file_path'])

X = np.stack(centroids)
print('X.shape=',X.shape)

# Normalize pixel values to [0, 1]
X = X / 255.0
y = 0 # just a patch to make things run, as we do not use labels in this example
#av: TODO replace compressed_set with point_set
# define set of points on which we will work
point_set = setofpoints.SetOfPoints(points=X)

#av: TODO choose landmarks randoomly or greeedily
# Special for MNist : 
# Select one sample per digit to serve as a landmark
landmarks = []
import random
for digit in range(3):
	landmarks.append(landmark.Landmark(random.randint(0, centroids.shape[0]),1.0))

mnist_problem = problem.Problem(point_set, r=1)
# mnist_problem.optimize(landmarks, k=4, target_avg_voltage=0.5)

# Initialize the map
voltage_map = voltagemap.VoltageMap()

# Compute voltages for each landmark and store in the map
for lm in landmarks:
	mnist_solver = solver.Solver(problem=mnist_problem)
	voltages = mnist_solver.compute_voltages(k=4, landmarks=[lm])
	voltage_map.add_solution(landmark_index=lm.index, voltages=voltages)

print(np.array(voltage_map.voltage_maps).shape)

#av: TODO call some visualizations, that store the figure into a file.
visualization.Visualization.plot_mds_unlabeled(voltage_map, point_set, alpha_actual=0.5, out_file="../inputoutput/matplotfigures/mnist_mds.png")