# Standard library imports
import os
from typing import Union, Optional, List, Any, Tuple

# Third-party imports
import numpy as np
from sklearn.datasets import fetch_openml
import faiss

# Local imports - core functionality
import landmark
import voltagemap
import problem
import solver
import setofpoints
import kmeans

# Local imports - utilities
from Utilities import config
from Utilities.distances import compute_distances

# Optional imports for visualization (commented out until needed)
# import matplotlib.pyplot as plt
# from sklearn.manifold import MDS
# from sklearn.decomposition import PCA

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
	# Setup directory structure
	import os

	# Get the directory containing the script
	script_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.dirname(script_dir)  # Go up one level to VoltageDimentionalReduction

	# Define all directory paths relative to project root
	data_dir = os.path.join(project_root, "Voltage_Data", "mnist")
	temp_dir = os.path.join(project_root, "Voltage_Temp")
	results_dir = os.path.join(temp_dir, "Results")
	intermediates_dir = os.path.join(temp_dir, "Intermediates")

	print(f"Project root: {project_root}")
	print(f"Looking for MNIST data in: {data_dir}")

	# Create necessary directories
	os.makedirs(data_dir, exist_ok=True)
	os.makedirs(results_dir, exist_ok=True)
	os.makedirs(intermediates_dir, exist_ok=True)

	# Check if MNIST data exists, download if not
	mnist_file = os.path.join(data_dir, "mnist.csv")
	if not os.path.exists(mnist_file):
		print(f"MNIST data file not found at {mnist_file}")
		print("Downloading MNIST dataset...")
		try:
			# Load MNIST dataset using scikit-learn
			X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
			
			# Save as CSV
			print("Saving MNIST dataset...")
			os.makedirs(os.path.dirname(mnist_file), exist_ok=True)
			data = np.column_stack((X, y))  # Combine features and labels
			np.savetxt(mnist_file, data, delimiter=',')
			print(f"MNIST dataset saved to {mnist_file}")
		except Exception as e:
			print(f"Error downloading MNIST dataset: {e}")
			print("Please ensure you have internet connection or manually place mnist.csv in the correct location.")
			exit(1)

	# Load configuration parameters
	config.params['file_path'] = mnist_file
	config.params['split_char'] = ','
	config.params['normalize_vecs'] = False

	config.params['max_centroids'] = 1000
	config.params['init_size'] = 5000
	config.params['batch_size'] = 1000
	config.params['kmeans_output'] = os.path.join(results_dir, 'streaming_centroids.npy')
	config.params['Voltage_map_output'] = os.path.join(results_dir, 'voltage_map.npy')
	config.params['k'] = 10

	run_kmeans=True  # Set to True to generate initial workspace

	# Set workspace file path
	workspace_file = os.path.join(intermediates_dir, "pointset.pkl")
	
	# Check for required packages
	try:
		import dill  # dill is used to save the workspace
	except ImportError:
		print("Error: The 'dill' package is required but not installed.")
		print("Please install it using: pip install dill")
		exit(1)

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
		try:
			dill.dump_session(workspace_file)
			print(f"Saved workspace to {workspace_file}")
		except Exception as e:
			print(f"Warning: Could not save workspace: {e}")
	else:
		try:
			if not os.path.exists(workspace_file):
				print(f"Error: Workspace file {workspace_file} not found!")
				print("Please run with run_kmeans=True first to generate the workspace file.")
				exit(1)
			dill.load_session(workspace_file)
			print(f"Loaded workspace from {workspace_file}")
		except Exception as e:
			print(f"Error loading workspace: {e}")
			print("Please run with run_kmeans=True first to generate the workspace file.")
			exit(1)
	
	print("starting building landmarks after kmeans is done")

	# Initialize the map
	voltage_map = voltagemap.VoltageMap()
	first_entry = all_voltages.entries[0]  # get the first landmark and its voltages
	voltage_map.add_solution(first_entry['landmark'], voltages=first_entry['voltages'])
	max_voltage = np.zeros(len(all_voltages))  # to keep track of the maximum voltage for each landmark

# repeatedly iteration all_voltages.entries and add the landmark with the largest distance to the selected landmarks to the voltage map	
	for iteration in range(100):
		# Find the landmark in all_voltages.entries that is farthest from the current voltage_map entries
		max_min_dist = 2.0
		best_idx = None
		best_norm = 2.0
		for idx, entry in enumerate(all_voltages.entries):
			# Skip if already in voltage_map
			if any(np.array_equal(entry['landmark'].index, vmap_entry['landmark'].index) 
				   for vmap_entry in voltage_map.entries):
				continue	
			# Compute minimum distance to any entry in voltage_map
			min_dist = np.min([np.linalg.norm(entry['voltages'] - vmap_entry['voltages']) 
							  for vmap_entry in voltage_map.entries])
			if min_dist > max_min_dist and entry['advantage'] > best_norm:
				max_min_dist = min_dist
				best_idx = idx
				best_norm = entry['advantage']
		print(f"Iteration {iteration}: Best landmark index {best_idx} norm={best_norm:.4f} with min distance {max_min_dist:.4f}")
		if best_idx is not None:
			entry = all_voltages.entries[best_idx]
			voltage_map.add_solution(entry['landmark'], voltages=entry['voltages'])
		else:
			break

	print("Applying voltage-based filtering")
	from filter import filter_by_voltage, filter_by_weights

	# Try different threshold values until we keep enough points
	thresholds = [1.0, 0.7, 0.5, 0.3, 0.1]  # Start strict, gradually relax
	min_maps_values = [2, 1]  # Try different min_maps values
	filtered_points = None

	filter_indices = None  # Store indices of filtered points
	for threshold in thresholds:
		for min_maps in min_maps_values:
			filtered_points, filter_mask = filter_by_voltage(
				voltage_map=voltage_map,
				point_set=centroids,
				threshold=threshold,
				min_maps=min_maps
			)
			print(f"Threshold={threshold}, min_maps={min_maps}: "
				  f"kept {len(filtered_points)} points out of {len(centroids)}")
			
			# If we have enough points, break out of both loops
			if len(filtered_points) >= 100:  # Minimum desired number of points
				print(f"Found good filtering parameters: threshold={threshold}, min_maps={min_maps}")
				filter_indices = np.where(filter_mask)[0]  # Store indices where mask is True
				break
		if len(filtered_points) >= 100:
			break

	# If filtering was too strict, use original points
	if len(filtered_points) < 100:
		print("Warning: Filtering was too strict, using original points")
		filtered_points = centroids

	print(f"\nFinal filtered set: {len(filtered_points)} points")

	print("Computing distances on filtered points")
	if len(filtered_points) > 0:
		# Convert filtered points to numpy array format
		if isinstance(filtered_points, setofpoints.SetOfPoints):
			points_array = filtered_points.points
		else:
			points_array = filtered_points

		print(f"Points array shape: {points_array.shape}")
		
		# Extract voltage maps as numpy array and filter to match points
		voltage_maps = voltage_map.all_solutions()
		# Take only the voltage maps for the filtered points using the saved indices
		if filter_indices is not None:
			voltage_maps = voltage_maps[filter_indices]
		print(f"Voltage maps shape: {voltage_maps.shape}")

		# Compute distances
		Ds = compute_distances(points_array, voltage_maps)
		print(f"len(Ds) = {len(Ds)}")
		print(f"Distance matrices shapes:")
		print(f"D1 (Euclidean in original space): {Ds[0].shape}")
		print(f"D2 (Euclidean in voltage space): {Ds[1].shape}")
		print(f"D3 (Graph-based): {Ds[2].shape}")
		Deuc = Ds[0]
		Dvolt = Ds[1]
		Dgraph = Ds[2]
	else:
		print("Error: No points to compute distances on")
		exit(1)
	
	# save the workspace for later use
	workspace_final = os.path.join(intermediates_dir, "workspace.pkl")
	dill.dump_session(workspace_final)
	print(f"Workspace saved to {workspace_final}")

	import pickle
	with open(config.params['Voltage_map_output'], 'wb') as f:
		pickle.dump(voltage_map, f)
	print(f"Voltage map saved to {config.params['Voltage_map_output']}")

