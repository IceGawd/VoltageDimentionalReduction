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
	all_voltages = voltagemap.VoltageMap()	
	_problem = problem.Problem(centroids,r=config.params['r'])
	_solver=solver.Solver(_problem)
	for index in range(len(centroids)):
		_landmark= landmark.Landmark(index, voltage=1.0)
		voltages=_solver.compute_voltages(_landmark)
		all_voltages.add_solution(_landmark, voltages=voltages)
	return all_voltages

def main():
	timer=Timer()
	# generate centroids using streaming k-means
	points, counters, majority_labels, _,_=kmeans.Streaming_Kmeans(config.params['file_path'])
	timer.mark("Streaming K-means completed")
	
	X=np.stack(points)
	y= np.array(majority_labels)

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
		'all_voltages': all_voltages,		# your VoltageMap object
		'centroids': centroids,				# your SetOfPoints object
		'k': config.params['k']				# number of neighbors
	}

	if ('Voltage_map_output' in config.params):
		with open(config.params['Voltage_map_output'], 'wb') as f:
			pickle.dump(data_to_save, f)
		print(f"Voltage map saved to {config.params['Voltage_map_output']}")

	return data_to_save

if __name__ == "__main__":
	from Utilities.set_params import set_params
	set_params()
	if config.params['test']:
		# Load configuration parameters
		config.params['file_path']= '../../Voltage_Data/glove/glove_with_pos.txt'
		config.params['split_char']= ' '
		config.params['normalize_vecs']= False
		config.params['kmeans_output']= '../../Voltage_Temp/Results/glove/streaming_centroids.npy'
		config.params['Voltage_map_output']= '../../Voltage_Temp/Results/glove/voltage_map.npy'

		# config.params['file_path']= '../../Voltage_Data/mnist/mnist.csv'
		# config.params['split_char']= ','
		# config.params['normalize_vecs']= False
		# config.params['kmeans_output']= '../../Voltage_Temp/Results/streaming_centroids.npy'
		# config.params['Voltage_map_output']= '../../Voltage_Temp/Results/voltage_map.npy'

		config.params['max_centroids']= 1000
		config.params['init_size']= 5000
		config.params['batch_size']= 1000
		config.params['k']=10

		main()
