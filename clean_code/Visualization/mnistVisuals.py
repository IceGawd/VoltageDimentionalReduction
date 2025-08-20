import matplotlib.pyplot as plt
import numpy as np
from typing import List

from Visualization import visualHelpers

def _prepare_image_rgba(digit_array, color, alpha_actual):
	alpha_mask = np.clip(digit_array.reshape(28, 28), 0, 255) / 255
	rgb_image = np.zeros((28, 28, 4))
	for c in range(3):
		rgb_image[..., c] = color[c]
	rgb_image[..., 3] = alpha_mask * alpha_actual
	return rgb_image

def _plot_digits(transformed_points, point_colors, data, voltages, ax, image_size, landmarkSize, alpha_actual, remove_clutter):
	landmark_indicies = [landmark.index for landmark in voltages.get_all_landmarks()]
	drawn_xy = []
	size_sqrd = image_size ** 2
	for i in range(transformed_points.shape[0]):
		color = point_colors[i]
		size = landmarkSize if i in landmark_indicies else 1
		rgba_img = _prepare_image_rgba(data[i], color, alpha_actual)
		x, y = transformed_points[i]

		draw = True

		if remove_clutter:
			for (x2, y2) in drawn_xy:
				if ((x2 - x) ** 2 + (y2 - y) ** 2 < size_sqrd):
					draw = False
					break

		if draw:
			if remove_clutter:
				drawn_xy.append((x, y))

			ax.imshow(
				rgba_img,
				extent=(x - image_size * size, x + image_size * size, y - image_size * size, y + image_size * size),
				origin='upper'
			)

def plot_mnist_unlabeled(voltages, data, transformation="mds", landmarkSize=3, alpha_actual=1, percent_size=0.02, argmax=True, remove_clutter=True, out_file=None):
	"""
	Visualizes MNIST digits in 2D space after dimensionality reduction (MDS or PCA),
	coloring and sizing them based on their voltage values.
	"""
	points = voltages.voltage_array()
	transformed_points = visualHelpers.transform(points, transformation)
	x_bound, y_bound, image_size = visualHelpers.compute_image_size(transformed_points, percent_size)
	fig, ax = visualHelpers.setup_figure(x_bound, y_bound, image_size, landmarkSize, "Visualization of K-Means MNIST")

	colors = visualHelpers.get_distinct_colors(points[0].shape[0])
	point_colors = [colors[np.argmax(p)] for p in points] if argmax else [colors[np.argmin(p)] for p in points]

	_plot_digits(transformed_points, point_colors, data, voltages, ax, image_size, landmarkSize, alpha_actual, remove_clutter)
	visualHelpers.standard_save_display(out_file)

def plot_mnist_digits(voltages, data, labels, transformation="mds", landmarkSize=3, alpha_actual=1, percent_size=0.02, remove_clutter=True, out_file=None):
	"""
	Visualizes MNIST digits in 2D space using voltage-based embeddings reduced by PCA or MDS.
	Each digit is rendered as a translucent RGB image, colored by its true label.
	"""
	points = voltages.voltage_array()
	transformed_points = visualHelpers.transform(points, transformation)
	x_bound, y_bound, image_size = visualHelpers.compute_image_size(transformed_points, percent_size)
	fig, ax = visualHelpers.setup_figure(x_bound, y_bound, image_size, landmarkSize, "Visualization of Digits")

	colors = visualHelpers.get_distinct_colors(len(set(labels)))
	point_colors = [colors[int(l)] if l is not None else (1, 1, 1) for l in labels]

	_plot_digits(transformed_points, point_colors, data, voltages, ax, image_size, landmarkSize, alpha_actual, remove_clutter)
	visualHelpers.standard_save_display(out_file)
