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

import kmeans
import landmark
import voltagemap
import problem
import solver
import visualization
import setofpoints
# yoav: TODO get rid of number of cores problem
# yoav: TODO: read points using streaming Kmeans

## Loading mnist data, no kmeans
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(np.int64)

# Normalize pixel values to [0, 1]
X = X / 255.0

# define set of points on which we will work
point_set = setofpoints.SetOfPoints(points=X[:1000])

#av: TODO choose landmarks randoomly or greeedily
# Special for MNist : 
# Select one sample per digit to serve as a landmark
landmarks = []
for digit in range(10):
	indices = np.where(y == digit)[0]
	# Choose the first occurrence as the landmark
	landmarks.append(landmark.Landmark.createLandmarkClosestTo(point_set, X[indices[0]], 1))

mnist_problem = problem.Problem(point_set, r=1)
mnist_problem.optimize(landmarks, k=2, r=0.5)

# Initialize the map
voltage_map = voltagemap.VoltageMap()

# Compute voltages for each landmark and store in the map
for lm in landmarks:
	mnist_solver = solver.Solver(problem=mnist_problem)
	voltages = mnist_solver.compute_voltages(k=2, landmarks=[lm])
	voltage_map.add_solution(landmark_index=lm.index, voltages=voltages)

#av: TODO call some visualizations, that store the figure into a file.
visualization.Visualization.plot_mds_digits([2, 3, 4, 5, 7, 8, 9], voltage_map, point_set, y[:1000], alpha_actual=0.5, out_file="../inputoutput/matplotfigures/mnist_mds.png")