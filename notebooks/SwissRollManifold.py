#!/usr/bin/env python
# coding: utf-8

import sys
import os
import importlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from sklearn.datasets import make_swiss_roll

# Local imports
sys.path.append(os.path.abspath("../clean_code/"))
from Visualization import generalVisualization, gloveVisuals, visualHelpers
from Utilities import config
import select_landmarks_MI
import voltagemap
import setofpoints
import problem
import solver
import landmark
import main

# Reload in case modules are being iterated on interactively
importlib.reload(generalVisualization)
importlib.reload(gloveVisuals)
importlib.reload(visualHelpers)
importlib.reload(voltagemap)
importlib.reload(main)

# Configure parameters
config.params["k"] = 2
config.params["r"] = 1
config.params["NoOfLandmarks"] = 10
config.params["DepthOfLandmarkSearch"] = 3

# --- Data generation ---
def generate_swissroll_setofpoints(n_samples=1000, noise=0.0) -> setofpoints.SetOfPoints:
	"""
	Generate a Swiss roll dataset and wrap it in a SetOfPoints object.

	Parameters
	----------
	n_samples : int
		Number of data points.
	noise : float
		Standard deviation of Gaussian noise.

	Returns
	-------
	SetOfPoints
		The generated dataset with uniform weights.
	"""
	points, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
	return setofpoints.SetOfPoints(points)


# --- Voltage computation ---
def compute_voltages_with_landmarks(points, landmark_indices) -> voltagemap.VoltageMap:
	"""
	Compute the voltage map for specific landmark indices.

	Parameters
	----------
	points : SetOfPoints
		Dataset containing the points.
	landmark_indices : list[int]
		Indices of landmarks to compute voltages for.

	Returns
	-------
	VoltageMap
		The computed voltage map.
	"""
	vmap = voltagemap.VoltageMap(solver.Solver)
	prob = problem.Problem(points, r=config.params["r"])
	sol = solver.Solver(prob)

	for idx in landmark_indices:
		lm = landmark.Landmark(idx, voltage=1.0)
		voltages = sol.compute_voltages(lm)
		vmap.add_entry([idx], voltages)

	return vmap

# --- Main workflow ---
if __name__ == "__main__":
	# Generate dataset
	centroids = generate_swissroll_setofpoints()

	"""
	# Pick closest 10 points to origin as landmarks
	distances = np.linalg.norm(centroids, axis=1)
	closest_indices = np.argsort(distances)[:10]
	"""

	allVoltages = main.compute_voltages(centroids)
	voltage_map = select_landmarks_MI.select_landmarks(allVoltages)

	# Log-transform voltages for visualization
	log_vmap = voltagemap.VoltageMap(solver.Solver)
	for entry in voltage_map.entries:
		log_vmap.add_entry(entry["landmark"], np.log(entry["voltages"]))

	# Plot results
	generalVisualization.plot_3d_voltage_colored(log_vmap, centroids, out_file="../../Voltage_Data/images/mnist_visualization.png")
