import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from typing import List
from collections import Counter
import math

import setofpoints
from Visualization import visualHelpers
from Utilities import voltage_embed
from Utilities import shuffle

def _prepare_image_rgba(digit_array, color, alpha_actual):
	alpha_mask = np.clip(digit_array.reshape(28, 28), np.min(digit_array), np.max(digit_array)) / (np.max(digit_array) - np.min(digit_array)) + np.min(digit_array)
	rgb_image = np.zeros((28, 28, 4))
	for c in range(3):
		rgb_image[..., c] = color[c]
	rgb_image[..., 3] = alpha_mask * alpha_actual
	return rgb_image

def _plot_digits(point_transformed_voltages, point_colors, data, voltages, ax, image_size, landmarkSize, alpha_actual, remove_clutter):
	landmark_indicies = [landmark.index for landmark in voltages.get_all_landmarks()]
	drawn_xy = []
	size_sqrd = image_size ** 2
	for i in range(point_transformed_voltages.shape[0]):
		color = point_colors[i]
		size = landmarkSize if i in landmark_indicies else 1
		rgba_img = _prepare_image_rgba(data[i], color, alpha_actual)
		x, y = point_transformed_voltages[i]

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


def plot_mnist_unlabeled(voltages, data, transformation="mds", landmarkSize=3, alpha_actual=1,
						 percent_size=0.02, argmax=True, remove_clutter=True, out_file=None):
	"""
	Visualizes MNIST digits in 2D space after dimensionality reduction (MDS or PCA),
	coloring and sizing them based on their voltage values.
	"""
	point_voltages = voltages.voltage_array()
	point_transformed_voltages = visualHelpers.transform(point_voltages, transformation)
	x_bound, y_bound, image_size = visualHelpers.compute_image_size(point_transformed_voltages, percent_size)
	fig, ax = visualHelpers.setup_figure(x_bound, y_bound, image_size, landmarkSize, "Visualization of K-Means MNIST")

	colors = visualHelpers.get_colors(point_voltages[0].shape[0])
	point_colors = [colors[np.argmax(p)] for p in point_voltages] if argmax else [colors[np.argmin(p)] for p in point_voltages]

	_plot_digits(point_transformed_voltages, point_colors, data, voltages, ax, image_size, landmarkSize, alpha_actual, remove_clutter)
	visualHelpers.standard_save_display(out_file)


def plot_mnist_digits(voltages, data, labels, transformation="mds", landmarkSize=3, alpha_actual=1,
					  percent_size=0.02, remove_clutter=True, out_file=None):
	"""
	Visualizes MNIST digits in 2D space using voltage-based embeddings reduced by PCA or MDS.
	Each digit is rendered as a translucent RGB image, colored by its true label.
	"""
	point_voltages = voltages.voltage_array()
	point_transformed_voltages = visualHelpers.transform(point_voltages, transformation)
	x_bound, y_bound, image_size = visualHelpers.compute_image_size(point_transformed_voltages, percent_size)
	fig, ax = visualHelpers.setup_figure(x_bound, y_bound, image_size, landmarkSize, "Visualization of Digits")

	colors = visualHelpers.get_colors(len(set(labels)))
	point_colors = [colors[int(l)] if l is not None else (1, 1, 1) for l in labels]

	_plot_digits(point_transformed_voltages, point_colors, data, voltages, ax, image_size, landmarkSize, alpha_actual, remove_clutter)
	visualHelpers.standard_save_display(out_file)


def compute_labels(label_counts, ratio_threshold=0.6, size_threshold=5, **kwargs):
	"""
	Compute labels based on the label counts, ratio threshold, and size threshold.
	"""
	str_labels = []
	for label_count in label_counts:
		if label_count is None:
			label = "small"
		else:
			common = label_count.most_common()
			total_count = sum([c[1] for c in common])
			ratio = common[0][1] / total_count
			if total_count < size_threshold:
				label = "small"
			else:
				if ratio > ratio_threshold:
					label = common[0][0]
				else:
					label = "weak_maj"
		str_labels.append(label)
	return str_labels

def fontsize_for_data_height(ax, data_height):
	"""
	Compute a fontsize (in point_voltages) so that text drawn in data coords 
	roughly matches a given data height.
	"""
	fig = ax.get_figure()
	# Axis bounding box in display coordinates (pixels)
	bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
	ax_height_inch = bbox.height
	ax_height_data = ax.get_ylim()[1] - ax.get_ylim()[0]

	# How many data units per inch?
	data_per_inch = ax_height_data / ax_height_inch

	# Convert desired data height to inches, then to point_voltages (72 pt = 1 inch)
	return data_height / data_per_inch * 72

def radialCollisionDetector(collidable, drawn_circles, remove_clutter, force_draw, renderer, pad_pixels=2, **kwargs):
	"""
	Place collidable as a circle with collision handling.

	drawn_circles: list of (center, radius, collidable) tuples
	force_draw:    if True, draw this collidable and hide any overlapping others.
	"""

	# Get the bounding box of the image in display coords
	bbox = collidable.get_window_extent(renderer=renderer)

	# Center of the box
	cx = (bbox.x0 + bbox.x1) / 2
	cy = (bbox.y0 + bbox.y1) / 2

	# Radius = half the diagonal of the bbox
	width = bbox.width
	height = bbox.height
	radius = math.sqrt(width**2 + height**2) / 2 + pad_pixels

	if not remove_clutter:
		drawn_circles.append(((cx, cy), radius, collidable))
		return

	def circles_overlap(c1, r1, c2, r2):
		dx = c1[0] - c2[0]
		dy = c1[1] - c2[1]
		return dx*dx + dy*dy <= (r1 + r2)**2

	if force_draw:
		# bulldozer mode: keep this, hide others that collide
		survivors = []
		for (oc, orad, other_collidable) in drawn_circles:
			if circles_overlap((cx, cy), radius, oc, orad):
				other_collidable.set_visible(False)
			else:
				survivors.append((oc, orad, other_collidable))
		drawn_circles[:] = survivors
		drawn_circles.append(((cx, cy), radius, collidable))
	else:
		# normal mode: hide this if it collides with existing
		for (oc, orad, _) in drawn_circles:
			if circles_overlap((cx, cy), radius, oc, orad):
				collidable.set_visible(False)
				return
		drawn_circles.append(((cx, cy), radius, collidable))

def collisionDetector(collidable, drawn_boxes, remove_clutter, force_draw, renderer, pad_pixels=2, **kwargs):
	"""
	Draw collidable on an axes with collision handling.

	drawn_boxes: list of (bbox, collidable) tuples in DISPLAY coordinates.
	force_draw:  if True, draw this collidable and hide any overlapping others.
	"""
	this_bbox = collidable.get_window_extent(renderer=renderer).padded(pad_pixels)

	if not remove_clutter:
		drawn_boxes.append((this_bbox, collidable))
		return

	if force_draw:
		# bulldozer mode: hide others that collide, keep this one
		survivors = []
		for bbox, other_collidable in drawn_boxes:
			if this_bbox.overlaps(bbox):
				other_collidable.set_visible(False)
			else:
				survivors.append((bbox, other_collidable))
		drawn_boxes[:] = survivors  # overwrite survivors only
		drawn_boxes.append((this_bbox, collidable))
	else:
		# normal mode: hide this if it collides with existing
		for bbox, _ in drawn_boxes:
			if this_bbox.overlaps(bbox):
				collidable.set_visible(False)
				return
		drawn_boxes.append((this_bbox, collidable))

def scatter_plot(point_voltages, point_transformed_voltages, data, focus_on, labels, reverse_dict_labels, 
				 percent_size=0.01, alpha_actual=1, out_file=None, element="digit", centroid_others=None, label_counts=None, dpi=100, remove_clutter=False, continous_label=False, show_legend=True, **kwargs):
	"""
	Creates a scatter plot of transformed point_voltages with digit images or point_voltages.
	
	Args:
		point_voltages: Voltage values for each point
		point_transformed_voltages: 2D transformed coordinates for plotting
		data: Original data points
		focus_on: Indices to focus on
		labels: Integer labels for each point
		reverse_dict_labels: Dictionary mapping integer labels back to string labels
		percent_size: Size of elements as percentage of plot area
		alpha_actual: Transparency level
		out_file: Output file path
		element: Type of elements to plot ("digit", "point", "label")
		centroid_others: Additional centroid data
		label_counts: Count data for labels
		dpi: Resolution for output
		remove_clutter: Whether to remove overlapping elements
		continous_label: Whether to use continuous label coloring
		show_legend: Whether to display a legend mapping colors to string labels
		**kwargs: Additional keyword arguments
	"""

	if out_file == None:
		out_file = "mnist_visualization.png"

	fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi)

	colors = visualHelpers.get_colors(len(reverse_dict_labels))

	x_bound = (point_transformed_voltages[:, 0].min(), point_transformed_voltages[:, 0].max())
	y_bound = (point_transformed_voltages[:, 1].min(), point_transformed_voltages[:, 1].max())
	image_size = (x_bound[1] + y_bound[1] - x_bound[0] - y_bound[0]) * percent_size / 2

	ax.set_xlim(x_bound[0] - image_size, x_bound[1] + image_size)
	ax.set_ylim(y_bound[0] - image_size, y_bound[1] + image_size)
	ax.set_facecolor('black')
	fig.patch.set_facecolor('black')
	plt.title("Visualization of Digits")

	count_nones = 0

	fontsize = fontsize_for_data_height(ax, 10 * percent_size)

	drawn_boxes = []

	if continous_label:
		cmap = cm.magma
		norm = mcolors.Normalize(
			vmin=min(min([float(ck) for ck in c.keys()]) for c in label_counts if c is not None),
			vmax=max(max([float(ck) for ck in c.keys()]) for c in label_counts if c is not None)
		)

	for i in range(point_transformed_voltages.shape[0]):
		voltages = point_voltages[i, :]
		label = labels[i]
		force_draw = False

		x, y = point_transformed_voltages[i]

		if (np.min(voltages) == 0): # only works for log transform
			min_index = np.argmin(voltages)
			min_index = focus_on[min_index]
			plt.text(x, y, str(min_index), fontsize=20, color='white', ha='center', va='center')
			force_draw = True

		if continous_label and label_counts is not None and label_counts[i] is not None:
			counter = label_counts[i]
			total = sum(counter.values())
			avg_label = sum(float(k) * v for k, v in counter.items()) / total
			color = cmap(norm(avg_label))
		else:
			# --- discrete coloring ---
			if (label is not None) and (label != 0):
				color = np.array(colors[int(label)])
			else:
				count_nones += 1
				continue

		if label == 1:  # weak_maj
			plt.plot(x, y, marker='o', markersize=1, color=color)
		else:
			if element == "digit":
				rgb_image = _prepare_image_rgba(data[i], color, alpha_actual)
				image = ax.imshow(
					rgb_image,
					extent=(x - image_size, x + image_size, y - image_size, y + image_size),
					origin="upper"
				)
				radialCollisionDetector(image, drawn_boxes, remove_clutter, force_draw, fig.canvas.get_renderer(), **kwargs)

			elif element == "point":
				plt.plot(x, y, marker="o", markersize=6, color=color, alpha=alpha_actual)

			elif element == "label":
				text = ax.text(
					x, y, str(reverse_dict_labels[label]),
					color=color, fontsize=fontsize, alpha=alpha_actual,
					ha="center", va="center"
				)
				collisionDetector(text, drawn_boxes, remove_clutter, force_draw, fig.canvas.get_renderer(), **kwargs)

			elif centroid_others is not None:
				if len(centroid_others[i]) > 0:
					counter = Counter(d[element] for d in centroid_others[i] if element in d)
					most_common_value, count = counter.most_common(1)[0]
					text = ax.text(
						x, y, str(most_common_value),
						color=color, fontsize=fontsize, alpha=alpha_actual,
						ha="center", va="center"
					)
					collisionDetector(text, drawn_boxes, remove_clutter, force_draw, fig.canvas.get_renderer(), **kwargs)
				else:
					plt.plot(x, y, marker="o", markersize=6, color=color)
			else:
				raise ValueError("element must be either 'digit', 'point' or 'label'")

	# Create legend mapping colors to string labels
	if show_legend and not continous_label:
		legend_elements = []
		for label_id, label_str in reverse_dict_labels.items():
			if label_id != 0:  # Skip 'small' labels as they are not plotted
				color = colors[int(label_id)]
				legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
												markerfacecolor=color, markersize=10, label=label_str))
		
		if legend_elements:
			# Sort legend elements by label for consistent ordering
			legend_elements.sort(key=lambda x: x.get_label())
			
			legend = ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
							 frameon=True, fancybox=True, shadow=True, ncol=1, fontsize=10,
							 title="Labels", title_fontsize=12)
			legend.get_frame().set_facecolor('white')
			legend.get_frame().set_alpha(0.9)
			legend.get_frame().set_edgecolor('gray')
			legend.get_title().set_color('black')

	print(f"Number of point_voltages with no label: {count_nones}")
	visualHelpers.standard_save_display(out_file)

def plot_landmark_subset(point_voltages, centroids, label_counts, focus_on=None, log_transform=True,
						 transformation='pca', centroid_others=None, **kwargs):
	"""
	Visualizes a subset of point_voltages in 2D space after dimensionality reduction, focusing on specific landmarks.
	"""

	if focus_on is None or focus_on==[]:
		focus_on = np.array(range(point_voltages.shape[1]))

	if log_transform:
		point_voltages = -np.log(point_voltages)

	str_labels = compute_labels(label_counts, **kwargs)

	possible_labels = set(str_labels) - set(['small', 'weak_maj'])
	dict_labels = {label: i + 2 for i, label in enumerate(sorted(list(possible_labels)))}
	dict_labels['small'] = 0
	dict_labels['weak_maj'] = 1
	labels = [dict_labels[label] for label in str_labels]
	reverse_dict_labels = {value: key for key, value in dict_labels.items()}

	closest_landmarks = np.argmin(point_voltages, axis=1)
	mask = np.isin(closest_landmarks, focus_on)

	point_voltages = point_voltages[mask, :]
	point_voltages = point_voltages[:, focus_on]
	labels = np.array(labels)[mask]

	if centroid_others:
		centroid_others = [c for c, keep in zip(centroid_others, mask) if keep]

	point_transformed_voltages = visualHelpers.transform(point_voltages, transformation)

	# print(point_voltages.shape)
	# print(point_transformed_voltages.shape)
	# print(len(centroids))
	# print(len(labels))
	# print(len(reverse_dict_labels))

	scatter_plot(point_voltages, point_transformed_voltages, centroids, focus_on, labels, reverse_dict_labels, centroid_others=centroid_others, label_counts=label_counts, **kwargs)

def plot_point_sample(X_data, y_data, other_data, point_voltages, centroids, voltage_map, label_counts, centroid_others, **kwargs):
	# Embed new features from the voltage map
	features = voltage_embed.embed_voltage_features(X_data, centroids, voltage_map)

	# Collect landmark data
	all_point_voltages = [point_voltages[landmark.index] for landmark in voltage_map.get_all_landmarks()]
	data = [centroids[landmark.index] for landmark in voltage_map.get_all_landmarks()]
	all_label_counts = [label_counts[landmark.index] for landmark in voltage_map.get_all_landmarks()]
	all_centroid_others = [centroid_others[landmark.index] for landmark in voltage_map.get_all_landmarks()]

	# print("all_point_voltages[0]: " + str(all_point_voltages[0]))
	# print("data[0]: " + str(data[0]))
	# print("all_label_counts[0]: " + str(all_label_counts[0]))

	# print("features[0]: " + str(features[0]))
	# print("X_data[0]: " + str(X_data[0]))
	# print("Counter({y_data[0]: 1}): " + str(Counter({y_data[0]: 1})))

	# Extend with full dataset
	all_point_voltages.extend(features)
	data.extend(X_data)
	all_centroid_others.extend([[d] for d in other_data])

	# Wrap y_data (numpy labels) into Counters so it matches label_counts structure
	all_label_counts.extend([Counter({label: 1}) for label in y_data])

	# Make sure arrays are numpy arrays where expected
	all_point_voltages = np.array(all_point_voltages)
	data = setofpoints.SetOfPoints(np.array(data))

	# Plot
	plot_landmark_subset(all_point_voltages, data, all_label_counts, centroid_others=all_centroid_others, **kwargs)

def plot_points_from_file(filepath, n_points, point_voltages, centroids, voltage_map, label_counts, centroid_others, **kwargs):
	X_data, y_data, other_data = shuffle.load_data(filepath, n_points)
	plot_point_sample(X_data, y_data, other_data, point_voltages, centroids, voltage_map, label_counts, centroid_others, **kwargs)