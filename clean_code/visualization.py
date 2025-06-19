import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from typing import List
from scipy.optimize import minimize

import voltagemap
import setofpoints

class Visualization:
	"""
	Visualization utilities for displaying voltage maps on 2D point sets.

	Primarily intended to visualize outputs from the Solver.
	"""

	@staticmethod
	def get_distinct_colors(N):
		def inverse_pairwise_squared_distance_sum(points):
			n_points = points.shape[0]
			dist_sum = 0.0
			for i in range(n_points):
				for j in range(i + 1, n_points):
					dist_sum += 1.0 / (np.sum((points[i] - points[j]) ** 2) + 10 ** (-len(points)))
			return dist_sum
		
		def objective(x, n):
			points = x.reshape((n, 3))
			# Add the origin manually
			all_points = np.vstack(([0, 0, 0], points))
			return inverse_pairwise_squared_distance_sum(all_points)  # Negate for minimization
		
		# Initial guess: random points in unit cube
		init_points = np.random.rand(N, 3)
		result = minimize(
			objective,
			init_points.flatten(),
			args=(N,),
			method='L-BFGS-B',
			bounds=[(0, 1)] * (3 * N),
			options={'maxiter': 1000}
		)
		optimized_points = result.x.reshape((N, 3))
		return optimized_points

	@staticmethod
	def plot_mds_unlabeled(voltages, data, landmarkSize=3, alpha_actual=1, percent_size=0.02, out_file=None):
		"""
		Draws the MNIST digits corresponding to each point after running MDS

		Args:
			voltages (VoltageMap): The digits to plot
			data (SetOfPoints): The digits to plot
			landmarkSize (Optional[float]): How much bigger should landmarks be?
			alpha_actual (Optional[float]): The opacity of each digit, 1 is fully opaque and 0 is fully transparent
			percent_size (Optional[float]): The size of each digit, 1 is the size of the whole space and 0 is no image size
			out_file (Optional[str]): If provided, the output path to save the figure (e.g., "digits.png")
		"""

		voltages = np.array(voltages.voltage_maps)
		points = np.array(list(map(list, zip(*voltages))))

		mds = MDS(n_components=2)
		transformed_points = mds.fit_transform(points)

		fig, ax = plt.subplots(figsize=(12, 10))

		colors = Visualization.get_distinct_colors(voltages.shape[0])
		
		x_bound = (transformed_points[:, 0].min(), transformed_points[:, 0].max())
		y_bound = (transformed_points[:, 1].min(), transformed_points[:, 1].max())

		image_size = (x_bound[1] + y_bound[1] - x_bound[0] - y_bound[0]) * percent_size / 2
		
		for i in range(transformed_points.shape[0]):
			alpha_mask = np.clip(data[i].reshape(28, 28), 0, 1)

			point_voltages = voltages[:, i]
			
			color = np.array(colors[np.argmax(point_voltages)])

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
		plt.title("MDS Visualization of K-Means MNIST")

		if out_file:
			plt.savefig(out_file)
			plt.close(fig)
		else:
			plt.show()
	
	@staticmethod
	def plot_mds_digits(selected_digits, voltages, data, correct, n_outliers=10, alpha_actual=1, percent_size=0.02, out_file=None):
		"""
		Draws the MNIST digits corresponding to each point after running MDS

		Args:
			selected_digits (List[int]): The digits to plot
			voltages (VoltageMap): The digits to plot
			data (SetOfPoints): The digits to plot
			correct (List[int]): The correct labels for each point in data
			n_outliers (Optional[int]): The number of outliers to remove
			alpha_actual (Optional[float]): The opacity of each digit, 1 is fully opaque and 0 is fully transparent
			percent_size (Optional[float]): The size of each digit, 1 is the size of the whole space and 0 is no image size
			out_file (Optional[str]): If provided, the output path to save the figure (e.g., "digits.png")
		"""

		voltages = np.array(voltages.voltage_maps)

		indices = [i for i, label in enumerate(correct) if label in selected_digits]
		filtered_voltages = voltages[np.ix_(selected_digits, indices)]
		points = np.array(list(map(list, zip(*filtered_voltages))))

		filtered_data = np.array([data[i] for i in indices])
		filtered_labels = np.array([correct[i] for i in indices])

		# Step 1: Run MDS on voltages    
		mds = MDS(n_components=2)
		transformed_points = mds.fit_transform(points)

		# Step 2: Remove outliers
		center = np.mean(transformed_points, axis=0)
		distances = np.linalg.norm(transformed_points - center, axis=1)
		outlier_indices = np.argsort(distances)[-n_outliers:]
		mask = np.ones(len(transformed_points), dtype=bool)
		mask[outlier_indices] = False
		inlier_points = transformed_points[mask]
		
		# Step 3: Plot MNIST images with colored alpha
		fig, ax = plt.subplots(figsize=(12, 10))

		# Assign distinct colors for each digit
		colors = Visualization.get_distinct_colors(len(selected_digits))
		
		digit_colors = {}

		for i, label in enumerate(selected_digits):
			digit_colors[label] = colors[i]

		x_bound = (inlier_points[:, 0].min(), inlier_points[:, 0].max())
		y_bound = (inlier_points[:, 1].min(), inlier_points[:, 1].max())

		image_size = (x_bound[1] + y_bound[1] - x_bound[0] - y_bound[0]) * percent_size / 2
		
		for i in np.where(mask)[0]:
			alpha_mask = np.clip(filtered_data[i].reshape(28, 28), 0, 1)

			label = filtered_labels[i]
			color = np.array(digit_colors[label])
			if (np.max(voltages[:, i]) > 0.9):
				color = np.array([1, 1, 1])

			# Create RGBA image
			rgb_image = np.zeros((28, 28, 4))
			for c in range(3):
				rgb_image[..., c] = color[c]
			rgb_image[..., 3] = alpha_mask * alpha_actual  # Alpha from pixel intensity

			x, y = transformed_points[i]
			ax.imshow(rgb_image, extent=(x - image_size, x + image_size, y - image_size, y + image_size), origin='upper')
		
		ax.set_xlim(x_bound[0] - image_size, x_bound[1] + image_size)
		ax.set_ylim(y_bound[0] - image_size, y_bound[1] + image_size)
		ax.set_facecolor('black')
		fig.patch.set_facecolor('black')
		plt.title("MDS Visualization of Digits: " + ", ".join(map(str, selected_digits)))

		if out_file:
			plt.savefig(out_file)
			plt.close(fig)
		else:
			plt.show()
