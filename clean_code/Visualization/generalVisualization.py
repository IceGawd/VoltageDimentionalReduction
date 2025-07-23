import matplotlib.pyplot as plt
import numpy as np

from Visualization import visualHelpers

def plot_centroids(centroids, counters, majority_labels, out_file=None):
	"""
	Plots centroids with their counters as sizes and majority labels as annotations.

	Args:
		centroids (np.ndarray): Array of centroids (shape: [num_centroids, dimensions]).
		counters (np.ndarray): Array of counters for each centroid (shape: [num_centroids]).
		majority_labels (list): List of majority labels for each centroid.
		out_file (str, optional): Path to save the plot. If None, the plot is displayed.
	"""
	if centroids.shape[1] != 2:
		raise ValueError("Centroids must have 2 dimensions for visualization.")

	# Create the plot
	plt.figure(figsize=(10, 10))

	# plt.scatter(centroids[:, 0], centroids[:, 1], s=counters * 10, c='blue', alpha=0.6, label='Centroids')

	# Annotate each centroid with its majority label
	for i, (x, y) in enumerate(centroids):
		label = majority_labels[i] if majority_labels[i] is not None else "None"
		plt.text(x, y, str(label), fontsize=12, ha='center', va='center', color='red')

	# Add labels and title
	plt.xlabel("X Coordinate")
	plt.ylabel("Y Coordinate")
	plt.xlim([-1.1, 1.1])
	plt.ylim([-1.1, 1.1])
	plt.title("Centroids Visualization with Counters and Majority Labels")
	plt.legend()

	visualHelpers.standard_save_display(out_file)

def plot_landmark_covariance(voltage_map, out_file=None):
	voltages = [entry['voltages'] for entry in voltage_map.entries]
	plt.imshow(np.cov(voltages))
	plt.title("Covariance of Voltages")
	visualHelpers.standard_save_display(out_file)