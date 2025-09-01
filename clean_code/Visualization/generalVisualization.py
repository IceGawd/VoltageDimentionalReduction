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

def plot_with_landmarks_colored(data_to_save, transformation="mds"):
	va = data_to_save['voltage_map'].voltage_array()
	centroids = visualHelpers.transform(va, transformation)
	all_voltages = data_to_save['all_voltages'].entries
	
	# Collect all landmark indices
	landmark_indices = {entry['landmark'].index for entry in all_voltages}

	plt.figure(figsize=(8, 6))
	for i, point in enumerate(centroids):
		if np.max(va[i]) == 1:
			plt.scatter(point[0], point[1], color='yellow', label='Landmark' if i == list(landmark_indices)[0] else "")
		else:
			plt.scatter(point[0], point[1], color='blue', s=10)

	plt.title("Centroid Points with Landmarks Highlighted")
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.legend()
	plt.grid(True)
	plt.show()

def plot_landmark_covariance(voltage_map, out_file=None):
	voltages = [entry['voltages'] for entry in voltage_map.entries]
	plt.imshow(np.cov(voltages))
	plt.title("Covariance of Voltages")
	visualHelpers.standard_save_display(out_file)

def plot_3d_voltage_colored(vmap, points, out_file=None):
	"""
	Plot 3D scatter of points colored by their maximum voltage.

	Parameters
	----------
	vmap : VoltageMap
		Voltage map containing solutions.
	points : np.ndarray
		Array of shape (n_samples, n_dims).
	"""
	va = vmap.get_voltage_array()

	if points.shape[1] < 3:
		raise ValueError("Need at least 3 dimensions after transformation for 3D plot.")

	voltages = np.max(va, axis=1)  # could use np.mean instead
	norm = plt.Normalize(vmin=np.min(voltages), vmax=np.max(voltages))
	colors = plt.cm.viridis(norm(voltages))

	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection="3d")
	ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=20)

	ax.set_title("3D Voltage-Transformed Visualization")
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	plt.tight_layout()
	visualHelpers.standard_save_display(out_file)