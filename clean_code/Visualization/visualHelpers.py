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

import colorsys

def generate_vivid_colors(n):
    """Generate N vivid RGB colors for visibility on black background."""
    C=[colorsys.hsv_to_rgb(h, 1.0, 1.0)  # hue ∈ [0, 1), full saturation and brightness
       for h in [i / n for i in range(n)]]
    return np.array(C)
