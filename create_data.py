import numpy as np
import json
import pickle
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats

def save_data_json(data, output_file):
	pass

def save_data_pickle(data, output_file):
	pass

def load_data_json(data, input_file):
	pass

def load_data_pickle(data, input_file):
	pass

file_function_pairs = [["json", save_data_json, load_data_json], ["pkl", save_data_pickle, load_data_pickle]]

def data_function(data, file, save_or_load):
	for ffp in file_function_pairs:
		if file[-len(ffp[0]):] == ffp[0]:
			ffp[save_or_load](data, file)

def save_data(data, output_file):
	data_function(data, output_file, 1)

def load_data(data, input_file):
	data_function(data, input_file, 2)

def create_dataset_line(output_file="line.json", start=0, end=1, points=10, seed=42):
	data = []
	random.seed(seed)

	for p in range(points):
		data.append(np.array([random.random() * (end - start) + start]))

	save_data(data, output_file)

def create_dataset_square_edge(output_file="square_edge.json", p1=(0,0), p2=(1,1), points=100, seed=42):
	data = []
	random.seed(seed)

	x_diff = p2[0] - p1[0]
	y_diff = p2[1] - p1[1]

	for p in range(points):
		r = random.random() * 4
		side = int(r)
		var = r - side

		x_side = side % 2
		y_side = side >> 1

		x_rev = 1 - x_side
		y_rev = 1 - y_side

		data.append(np.array([
			var * x_side * x_diff + x_rev * y_side * x_diff + p1[0],	# 1st addition factor: vary on the x axis. 2nd addition factor: x to p2[0]. 3rd addition factor: x to p1[0]
			var * x_rev * y_diff + x_side * y_rev * y_diff + p1[1]		# 1st addition factor: vary on the y axis. 2nd addition factor: y to p2[1]. 3rd addition factor: y to p1[1]
		]))

	save_data(data, output_file)

def create_dataset_square_fill(output_file="square_fill.json", p1=(0,0), p2=(1,1), points=1000, seed=42):
	data = []
	random.seed(seed)

	x_diff = p2[0] - p1[0]
	y_diff = p2[1] - p1[1]

	for p in range(points):
		x_rand = random.random()
		y_rand = random.random()

		data.append(np.array([x_diff * x_rand + p1[0], y_diff * y_rand + p2[0]]))

	save_data(data, output_file)

def create_dataset_eigth_sphere(output_file="eigth_sphere.json", radius=1, x_pos=True, y_pos=True, z_pos=True, points=1000, seed=42):
	data = []
	random.seed(seed)

	for p in range(points):
		z = random.random()						# Z value
		angleXY = math.pi * random.random() / 2	# Angle in the XY plane

		point = [radius * math.sqrt(1 - z**2) * math.cos(angleXY), radius * math.sqrt(1 - z**2) * math.sin(angleXY), radius * z]
		data.append(np.array(point))

	save_data(data, output_file)

def dimentional_variation(dimentions):
	z_vals = []
	for d in range(dimentions):
		z_vals.append(stats.norm.ppf(random.random()))

	return np.array(z_vals)

def varied_point(mean, std):
	return mean + std * dimentional_variation(len(mean))

def select_random(array):
	return array[int(len(array) * random.random())]

def create_dataset_strong_clusters(output_file="strong_clusters.json", internal_std=1, external_std=10, mean=[0, 0], clusters=10, points=100, seed=42):
	data = []
	random.seed(seed)

	np_mean = np.array(mean)

	cluster_centers = []
	for c in range(clusters):
		cluster_centers.append(varied_point(np_mean, external_std))

	for p in range(points):
		data.append(varied_point(select_random(cluster_centers), internal_std))

	save_data(data, output_file)

def plotPoints(points):
	size = len(points[0])

	x_coords = [point[0] for point in points]

	if (size > 1):
		y_coords = [point[1] for point in points]
		if (size > 2):
			z_coords = [point[2] for point in points]
	else:
		y_coords = [0 for point in points]

	fig = plt.figure()
	ax = fig.add_subplot(111)

	if (size == 3):
		ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o', label='Points')
		ax.set_zlabel('Z-axis')
	else:
		ax.scatter(x_coords, y_coords, c='r', marker='o', label='Points')

	ax.set_xlabel('X-axis')
	ax.set_ylabel('Y-axis')

	ax.legend()

	plt.show()

if __name__ == '__main__':
	create_dataset_line()
	create_dataset_square_edge()
	create_dataset_square_fill()
	create_dataset_eigth_sphere()
	create_dataset_strong_clusters()