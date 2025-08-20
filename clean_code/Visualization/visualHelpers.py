import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import math
import itertools
from Utilities import config

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
	
	# Initial guess: points in unit cube
	div_num = int(np.ceil(math.pow(N, 1.0/3.0)))
	divisions = [(i + 1.0) / div_num for i in range(div_num)]

	all_points = list(itertools.product(divisions, repeat=3))
	init_points = np.array(all_points[:N])

	result = minimize(
		objective,
		init_points.flatten(),
		args=(N,),
		bounds=[(0, 1)] * (3 * N),
	)
	
	optimized_points = result.x.reshape((N, 3))
	return optimized_points

def transform(points, transformation):
	if transformation == "mds":
		mds = MDS(n_components=2)
		return mds.fit_transform(points)
	elif transformation == "pca":
		pca = PCA(n_components=2)
		return pca.fit_transform(points)
	else:
		raise ValueError("transformation must be either \"pca\" or \"mds\"")

def standard_save_display(out_file):
	if out_file:
		plt.savefig(out_file)
		print(f"Plot saved to {out_file}")
	if not ('no-show-plots' in config.params and config.params['no-show-plots']):
		plt.show()

def compute_image_size(transformed_points, percent_size):
	x_bound = (transformed_points[:, 0].min(), transformed_points[:, 0].max())
	y_bound = (transformed_points[:, 1].min(), transformed_points[:, 1].max())
	range_sum = (x_bound[1] + y_bound[1] - x_bound[0] - y_bound[0])
	return (x_bound, y_bound, range_sum * percent_size / 2)

def setup_figure(x_bound, y_bound, image_size, landmarkSize, title):
	fig, ax = plt.subplots(figsize=(12, 10))
	ax.set_xlim(x_bound[0] - image_size * landmarkSize, x_bound[1] + image_size * landmarkSize)
	ax.set_ylim(y_bound[0] - image_size * landmarkSize, y_bound[1] + image_size * landmarkSize)
	ax.set_facecolor('black')
	fig.patch.set_facecolor('black')
	plt.title(title)
	return fig, ax