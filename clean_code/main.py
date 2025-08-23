from Utilities.timer import Timer
import numpy as np
from itertools import product
import pandas
import matplotlib.pyplot as plt

import importlib

import landmark
import voltagemap
import problem
import solver
import setofpoints
import kmeans
from Utilities import config
import faiss


def compute_voltages(centroids):
	""" compute the voltage map for each centroid """

	from Utilities.timer import Timer
	timer=Timer()	
	timer
	all_voltages = voltagemap.VoltageMap()	
	_problem = problem.Problem(centroids,r=config.params['r'])
	_solver=solver.Solver(_problem)
	for index in range(len(centroids)):
		_landmark= landmark.Landmark(index, voltage=1.0)
		voltages=_solver.compute_voltages(_landmark)
		all_voltages.add_solution(_landmark, voltages=voltages)
	timer.mark("Computed voltages for all centroids")
	return all_voltages

def main(filepath):
	timer=Timer()
	# generate centroids using streaming k-means
	points, counters, majority_labels, label_counts, rms=kmeans.Streaming_Kmeans(config.params['file_path'])
	timer.mark("Streaming K-means completed")
	
	#X=np.stack(points)
	#y= np.array(label_counts)

	# define set of set of centroids
	centroids = setofpoints.SetOfPoints(points=points, weights=counters)

	timer.mark("Compute voltages started")
	# compute voltages for each centroid
	
	all_voltages = compute_voltages(centroids)
	print(f"all_voltages.all_solutions().shape: {all_voltages.all_solutions().shape}")

	timer.mark(f"Computed voltages for {len(centroids)} centroids")

	#from select_landmarks_MI import select_landmarks
	#voltage_map=select_landmarks(all_voltages)
	#timer.mark("Selected landmarks for voltage map")

	# Ds = compute_distances(centroids, voltage_map)

	import pickle
	data_to_save = {
		'majority_labels': majority_labels,	# your labels
		'label_counts': label_counts,		# your label counts
		'all_voltages': all_voltages,		# your VoltageMap object
		'centroids': centroids				# your SetOfPoints object
	}


	with open(config.params['save_data'], 'wb') as f:
		pickle.dump(data_to_save, f)
	print(f"Voltage map saved to {config.params['save_data']}")

	return data_to_save

if __name__ == "__main__":
	#import faulthandler
	#faulthandler.enable()

	from Utilities.set_params import set_params
	set_params()
	filepath = config.params['file_path']
	import os
	if not os.path.exists(filepath):
		raise FileNotFoundError(f"Input file {filepath} does not exist.")

	main(filepath)
