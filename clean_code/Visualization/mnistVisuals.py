import matplotlib.pyplot as plt
import numpy as np
from typing import List

from Visualization import visualHelpers

###Yoav: I think the functionality of plot_mnist_unlabeled can be merged with scatter_plot.
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


def compute_labels(label_counts,ratio_threshold=0.6, size_threshold=1):
    """
    Compute labels based on the label counts, ratio threshold, and size threshold.
    Args:
        label_counts (list): List of Counter objects containing label counts.
        ratio_threshold (float): Threshold for the ratio of the most common label.
        size_threshold (int): Minimum size for a label to be considered valid.
    Returns:
        list: List of labels based on the computed criteria."""
    str_labels = []
    for label_count in label_counts:   # Print the first 10 label counts   
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

def scatter_plot(points,transformed_points, data, focus_on, labels, reverse_dict_labels,
				 percent_size=0.01, alpha_actual=1, out_file=None, element="digit"):
	"""
	Creates a scatter plot of transformed points with digit images.

	Args:
		points (np.ndarray): Original points defined by the voltage maps
		transformed_points (np.ndarray): 2D coordinates of points after dimensionality reduction.
		data (np.ndarray): The raw MNIST images (each as a 784-length array) corresponding to the digits.
		focus_on (List[int]): List of indices to translate column in points to landmark numbers.
		labels ([np.ndarray]): Labels for coloring points.
		reverse_dict_labels (dict): Dictionary mapping label indices back to original labels.
		percent_size (float): Relative size of digit images as a fraction of plot range.
		alpha_actual (float): Opacity of digit images (0.0 to 1.0).
		out_file (Optional[str]): If provided, saves the output figure to this file path.
		element (str): The type of element being visualized, e.g., "digit" or "point".
	"""
	if out_file is None:
		out_file = "mnist_visualization.png"

	fig, ax = plt.subplots(figsize=(12, 10))

	# Assign distinct colors for each digit
	from Visualization.visualHelpers import generate_vivid_colors
	colors = generate_vivid_colors(len(reverse_dict_labels))

	# Define the boundaries of the plot based on transformed points
	x_bound = (transformed_points[:, 0].min(), transformed_points[:, 0].max())
	y_bound = (transformed_points[:, 1].min(), transformed_points[:, 1].max())

	image_size = (x_bound[1] + y_bound[1] - x_bound[0] - y_bound[0]) * percent_size / 2

	count_nones=0		

# iterate over each point and plot the corresponding digit image / landmark number

	for i in range(transformed_points.shape[0]):
	

		point_voltages = points[i, :]

		label = labels[i]

		if (label != None) & (label != 0):
			
			size = 1
			color = np.array(colors[int(label)])  # color for the point based on its label
			x, y = transformed_points[i]

			if (np.min(point_voltages) == 0.0):	
				#If landmark mark it's index
				#find the index of the minimal voltage
				min_index = np.argmin(point_voltages)
				min_index=focus_on[min_index]
				#plot the landmark number in the current location defined by transformed_points[i]
				plt.text(x,y , str(min_index), fontsize=20, color='white', ha='center', va='center')

			if label==1:  # If the label is 'weak_maj'
				# Plot the point with a smaller size
				plt.plot(x, y, marker='o', markersize=1, color=color)
			else:
				if element=="digit":

					# Create RGBA image of mnist digit with alpha mask
					rgb_image = np.zeros((28, 28, 4))
					alpha_mask = np.clip(data[i].reshape(28, 28), 0, 255) / 255  #The mask defines the silhouette of the digit
			
					rgb_image[..., 0:3] = color
					rgb_image[..., 3] = alpha_mask * alpha_actual  # Alpha from pixel intensity

					ax.imshow(rgb_image, extent=(x - image_size * size, x + image_size * size, y - image_size * size, y + image_size * size), origin='upper')
				elif element=="point":
					plt.plot(x, y, marker='o', markersize=6, color=color)

				else:
					raise ValueError("element must be either 'digit' or 'point'")
		else:
			count_nones+=1
	# finish the plot
	ax.set_xlim(x_bound[0] - image_size , x_bound[1] + image_size)
	ax.set_ylim(y_bound[0] - image_size , y_bound[1] + image_size)
	ax.set_facecolor('black')
	fig.patch.set_facecolor('black')
	plt.title("Visualization of Digits")

	print(f"Number of points with no label: {count_nones}")

	visualHelpers.standard_save_display(out_file)



def plot_landmark_subset(points,centroids,label_counts, focus_on = None, log_transform=True, transformation='pca',**kwargs):
	"""Visualizes a subset of points in 2D space after dimensionality reduction, focusing on specific landmarks.
	Specificaly, we filter out points whos closest landmark is not in focus_on. 
	We then remove the voltage maps that do not correspond to the voltage map.

	Args:
		points (np.ndarray): Original points defined by the voltage maps
		centroids (np.ndarray): The centroids of the clusters
		label_counts: (List[Counter]): List of label counts for each point, used for coloring.
		focus_on (List[int]): List of landmark indices on which to focus.
		log_transform (bool): Whether to apply log transformation to the points.
		transformation (str): The type of transformation to apply (e.g., 'pca').
		**kwargs: Additional keyword arguments for customization.
	"""

	if focus_on is None: 
		focus_on = np.array(range(points.shape[1]),**kwargs)  
	if log_transform:
		points = -np.log(points)	

	str_labels = compute_labels(label_counts)
	possible_labels = set(str_labels) - set(['small', 'weak_maj'])
	dict_labels = {label: i+2 for i, label in enumerate(sorted(list(possible_labels)))}
	dict_labels['small'] = 0
	dict_labels['weak_maj'] = 1
	labels=[dict_labels[label] for label in str_labels]
	reverse_dict_labels={ value:key for key, value in dict_labels.items()}


	# identify the points for whom the closest landmark is in focus_on
	closest_landmarks=np.argmin(points,axis=1)
	mask = np.isin(closest_landmarks, focus_on)

	# remove far points and remove voltagemaps that do not belong to focus_on
	points = points[mask,:]
	points = points[:, focus_on] 

#We need to make sure that the row numbers of labels and centroids those of points.

	# Remove the far points from the labels	
	labels=np.array(labels)
	labels = labels[mask]
	# Remove the far points from the centroids (the images)
	centroids=centroids[mask,:]

	# perform the dimensionality reduction:
	transformed_points = visualHelpers.transform(points, transformation)

	scatter_plot(points, transformed_points, centroids, focus_on, labels, reverse_dict_labels, **kwargs)
