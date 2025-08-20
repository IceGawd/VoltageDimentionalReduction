#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append(os.path.abspath("../clean_code/"))


# In[30]:

from Visualization import generalVisualization
from Visualization import gloveVisuals
from Visualization import visualHelpers
from Utilities import config

import pickle
import select_landmarks_MI
import importlib
import voltagemap
import setofpoints
import problem
import solver
import landmark
import main
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.datasets import make_swiss_roll


# In[3]:


importlib.reload(generalVisualization)
importlib.reload(gloveVisuals)
importlib.reload(visualHelpers)
importlib.reload(voltagemap)
importlib.reload(main)


# In[6]:

config.params['k']=2
config.params['r']=1


def generate_swissroll_setofpoints(n_samples=1000, noise=0.0) -> setofpoints.SetOfPoints:
	"""
	Generates a Swiss roll dataset and returns it as a SetOfPoints instance.

	Args:
		n_samples (int): Number of data points to generate.
		noise (float): Standard deviation of Gaussian noise added to the data.

	Returns:
		SetOfPoints: An instance containing the Swiss roll points with uniform weights.
	"""
	points, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
	return setofpoints.SetOfPoints(points)


# In[8]:


centroids = generate_swissroll_setofpoints()

def compute_voltages_with_landmarks(centroids, landmark_indices):
    """ compute the voltage map for specific landmark indices """
    all_voltages = voltagemap.VoltageMap()
    _problem = problem.Problem(centroids, r=config.params['r'])
    _solver = solver.Solver(_problem)
    
    for index in landmark_indices:
        _landmark = landmark.Landmark(index, voltage=1.0)
        voltages = _solver.compute_voltages(_landmark)
        all_voltages.add_solution(_landmark, voltages=voltages)
    
    return all_voltages

distances = np.linalg.norm(centroids, axis=1)

closest_indices = np.argsort(distances)[:10]

voltage_map = compute_voltages_with_landmarks(centroids, closest_indices)

# In[20]:


log_voltage_map = voltagemap.VoltageMap()
for entry in voltage_map.entries:
    log_voltage_map.add_solution(entry["landmark"], np.log(entry["voltages"]))


# In[31]:


def plot_3d_voltage_colored(voltage_map, centroids):
    va = voltage_map.voltage_array()

    if centroids.shape[1] < 3:
        raise ValueError("Need at least 3 dimensions after transformation for 3D plot.")

    voltages = np.max(va, axis=1)  # or change to something else, like np.mean(va, axis=1)
    norm = plt.Normalize(vmin=np.min(voltages), vmax=np.max(voltages))
    colors = plt.cm.viridis(norm(voltages))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c=colors, s=20)

    ax.set_title("3D Voltage-Transformed Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


# In[32]:


plot_3d_voltage_colored(log_voltage_map, centroids)


# In[ ]:




