import matplotlib.pyplot as plt
import numpy as np
from typing import List

from Visualization import visualHelpers

def plot_mnist_unlabeled(voltages, data, transformation="mds", landmarkSize=3, alpha_actual=1, percent_size=0.02, out_file=None):
	"""
	Visualizes MNIST digits in 2D space after dimensionality reduction (MDS or PCA),
	coloring and sizing them based on their voltage values.

	Args:
		voltages (VoltageMap): Contains a voltage vector for each digit, used to determine color and landmark status.
		data (SetOfPoints): The raw MNIST images (each as a 784-length array) corresponding to the digits.
		transformation (str, optional): Dimensionality reduction method to use ("mds" or "pca"). Defaults to "mds".
		landmarkSize (float, optional): Multiplier for the size of landmark digits (those with max voltage == 1). Defaults to 3.
		alpha_actual (float, optional): Global opacity of each digit image (0 = fully transparent, 1 = fully opaque). Defaults to 1.
		percent_size (float, optional): Relative size of each digit image as a fraction of the plot dimensions. Defaults to 0.02.
		out_file (str, optional): If provided, saves the plot to the specified file path instead of displaying it.

	Returns:
		None
	"""

	points = voltages.voltage_array()

	transformed_points = visualHelpers.transform(points, transformation)

	fig, ax = plt.subplots(figsize=(12, 10))

	colors = visualHelpers.get_distinct_colors(points[0].shape[0])

	x_bound = (transformed_points[:, 0].min(), transformed_points[:, 0].max())
	y_bound = (transformed_points[:, 1].min(), transformed_points[:, 1].max())

	image_size = (x_bound[1] + y_bound[1] - x_bound[0] - y_bound[0]) * percent_size / 2
	
	for i in range(transformed_points.shape[0]):
		alpha_mask = np.clip(data[i].reshape(28, 28), 0, 255) / 255

		point_voltages = points[i]
		
		color = np.array(colors[np.argmax(point_voltages)])

		size = 2
		if (np.max(point_voltages) == 1):
			size = landmarkSize
		
		# Create RGBA image
		rgb_image = np.zeros((28, 28, 4))
		for c in range(3):
			rgb_image[..., c] = color[c]
		rgb_image[..., 3] = alpha_mask * alpha_actual  # Alpha from pixel intensity

		x, y = transformed_points[i]
		ax.imshow(rgb_image, extent=(x - image_size * size, x + image_size * size, y - image_size * size, y + image_size * size), origin='upper')
	
	ax.set_xlim(x_bound[0] - image_size * landmarkSize, x_bound[1] + image_size * landmarkSize)
	ax.set_ylim(y_bound[0] - image_size * landmarkSize, y_bound[1] + image_size * landmarkSize)
	ax.set_facecolor('black')
	fig.patch.set_facecolor('black')
	plt.title("Visualization of K-Means MNIST")

	visualHelpers.standard_save_display(out_file)

def plot_mnist_digits(voltages, data, labels, transformation="mds", landmarkSize=3, alpha_actual=1, percent_size=0.02, out_file=None, log_transform=True):
	"""
	Visualizes MNIST digits in 2D space using voltage-based embeddings reduced by PCA or MDS.

	Each point in the embedding is rendered as a translucent RGB digit image colored by its label.
	Landmark points (voltage = 1) are optionally scaled up to highlight their influence.

	Args:
		voltages (VoltageMap): The VoltageMap containing voltages per landmark for each point.
		data (SetOfPoints): The list or array of raw MNIST digit images (each image flattened to length 784).
		labels (List[int]): The ground truth labels for each point (0–9). Used for coloring.
		transformation (str): Dimensionality reduction method: "pca" or "mds" (default: "mds").
		landmarkSize (float): Scaling factor for landmark digits (default: 3).
		alpha_actual (float): Opacity of digit images (0.0 to 1.0, default: 1).
		percent_size (float): Relative size of digit images as a fraction of plot range (default: 0.02).
		out_file (Optional[str]): If provided, saves the output figure to this file path.
		log_transform (bool): If True (default), applies a log transform to the voltage values before visualization.
	"""

	points = voltages.voltage_array()
	if log_transform:
		points = -np.log(points)	

	transformed_points = visualHelpers.transform(points, transformation)
	
	fig, ax = plt.subplots(figsize=(12, 10))

	# Assign distinct colors for each digit
	colors = visualHelpers.get_distinct_colors(len(set(labels)))
	
	x_bound = (transformed_points[:, 0].min(), transformed_points[:, 0].max())
	y_bound = (transformed_points[:, 1].min(), transformed_points[:, 1].max())

	image_size = (x_bound[1] + y_bound[1] - x_bound[0] - y_bound[0]) * percent_size / 2
	
	for i in range(transformed_points.shape[0]):
		alpha_mask = np.clip(data[i].reshape(28, 28), 0, 255) / 255

		point_voltages = points[i]

		label = labels[i]

		if (label != None):
			color = np.array(colors[int(label)])
			
			size = 1
			if (np.max(point_voltages) == 1):
				size = landmarkSize
	
			# Create RGBA image
			rgb_image = np.zeros((28, 28, 4))
	
			for c in range(3):
				rgb_image[..., c] = color[c]
			rgb_image[..., 3] = alpha_mask * alpha_actual  # Alpha from pixel intensity
	
			x, y = transformed_points[i]
			ax.imshow(rgb_image, extent=(x - image_size * size, x + image_size * size, y - image_size * size, y + image_size * size), origin='upper')
	
	ax.set_xlim(x_bound[0] - image_size * landmarkSize, x_bound[1] + image_size * landmarkSize)
	ax.set_ylim(y_bound[0] - image_size * landmarkSize, y_bound[1] + image_size * landmarkSize)
	ax.set_facecolor('black')
	fig.patch.set_facecolor('black')
	plt.title("Visualization of Digits")

	visualHelpers.standard_save_display(out_file)