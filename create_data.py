import numpy as np
import json
import ijson
import pickle
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
import time

def select_random(array):
	return array[int(len(array) * random.random())]

class Data():
	def __init__(self):
		self.data = None

	def __init__(self, arg):
		if isinstance(arg, list):
			self.data = np.array(arg)
		elif isinstance(arg, str):
			self.load_data(arg)
		else:
			self.data = nparray

	def __init__(self, input_file, streaming):
		if (streaming):
			self.data = stream_data_json(input_file)
		else:
			self.data = load_data(input_file)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		return self.data[index]

	def __setitem__(self, index, value):
		self.data[index] = value

	def save_data_json(self, output_file):
		with open(output_file, "w") as f:
			json.dump(np.array(self.data).tolist(), f)

	def save_data_pickle(self, output_file):
		with open(output_file, 'wb') as f: 
			pickle.dump(self.data, f) 

	def load_data_json(self, input_file):
		with open(input_file, 'r') as f:
			self.data = json.load(f)
			for i, point in enumerate(data):
				data[i] = np.array(point)

	def load_data_pickle(input_file):
		with open(input_file, 'r') as f:
			self.data = pickle.load(f)

	def stream_data_json(input_file):
		with open(input_file, 'r') as file:
			for point in ijson.items(file, 'item'):
				yield point

	file_function_pairs = [["json", save_data_json, load_data_json], ["pkl", save_data_pickle, load_data_pickle]]

	def data_function(self, file, save_or_load):
		if (file == None):
			return

		for ffp in self.file_function_pairs:
			if file[-len(ffp[0]):] == ffp[0]:
				if save_or_load == 1:
					ffp[save_or_load](self.data, file)
				else:
					return ffp[save_or_load](file)

	def save_data(self, output_file):
		self.data_function(output_file, 1)

	def load_data(self, input_file):
		self.data_function(input_file, 2)

	def get_random_point(self):
		return select_random(self.data)

def create_dataset_line(output_file="line.json", start=0, end=1, points=1000, seed=42):
	data = []
	random.seed(seed)

	for p in range(points):
		data.append(np.array([random.random() * (end - start) + start]))

	data = Data(data)
	data.save_data(output_file)

	return data

def create_dataset_square_edge(output_file="square_edge.json", p1=(0,0), p2=(1,1), points=1000, seed=42):
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

		variation = np.array([var * x_side * x_diff, var * x_rev * y_diff]) 	# Variations on the axis that draw the lines
		side = np.array([x_rev  * y_side * x_diff, x_side * y_rev * y_diff])	# Move the line to the side it is drawing on
		shift = np.array([p1[0], p1[1]])										# The shift to make the bottom left be p1

		data.append(np.array(variation + side + shift))

	data = Data(data)
	data.save_data(output_file)

	return data

def create_dataset_square_fill(output_file="square_fill.json", p1=(0,0), p2=(1,1), points=1000, seed=42):
	data = []
	random.seed(seed)

	x_diff = p2[0] - p1[0]
	y_diff = p2[1] - p1[1]

	for p in range(points):
		x_rand = random.random()
		y_rand = random.random()

		data.append(np.array([x_diff * x_rand + p1[0], y_diff * y_rand + p2[0]]))

	data = Data(data)
	data.save_data(output_file)

	return data

def create_dataset_eigth_sphere(output_file="eigth_sphere.json", radius=1, x_pos=True, y_pos=True, z_pos=True, points=1000, seed=42):
	data = []
	random.seed(seed)

	for p in range(points):
		z = random.random()						# Z value
		angleXY = np.pi * random.random() / 2	# Angle in the XY plane

		point = [radius * np.sqrt(1 - z**2) * np.cos(angleXY), radius * np.sqrt(1 - z**2) * np.sin(angleXY), radius * z]
		data.append(np.array(point))

	data = Data(data)
	data.save_data(output_file)

	return data

def dimentional_variation(dimentions):
	z_vals = []
	for d in range(dimentions):
		z_vals.append(stats.norm.ppf(random.random()))

	return np.array(z_vals)

def varied_point(mean, std):
	return mean + std * dimentional_variation(len(mean))

def create_dataset_strong_clusters(output_file="strong_clusters.json", internal_std=1, external_std=10, mean=[0, 0], clusters=10, points=1000, seed=42):
	data = []
	random.seed(seed)

	np_mean = np.array(mean)

	cluster_centers = []
	for c in range(clusters):
		cluster_centers.append(varied_point(np_mean, external_std))

	for p in range(points):
		data.append(varied_point(select_random(cluster_centers), internal_std))

	data = Data(data)
	data.save_data(output_file)

	return data

def rotate_into_dimention(data, higher_dim=3, seed=42):
	rotation_matrix = np.identity(higher_dim)



	if (seed != -1):
		random.seed(seed)

	for x1 in range(0, higher_dim - 1):
		for x2 in range(x1 + 1, higher_dim):
			angle = 2 * np.pi * random.random()

			rotateOnTheseAxes = np.identity(higher_dim)
			rotateOnTheseAxes[x1, x1] = np.cos(angle)
			rotateOnTheseAxes[x2, x2] = np.cos(angle)
			rotateOnTheseAxes[x1, x2] = np.sin(angle)
			rotateOnTheseAxes[x2, x1] = -np.sin(angle)

			rotation_matrix = np.matmul(rotation_matrix, rotateOnTheseAxes)

	print(rotation_matrix)

	data.data = list(data.data)

	for i in range(0, len(data)):
		extendedPoint = np.zeros(higher_dim)
		extendedPoint[:len(data[i])] = data[i]

		data[i] = np.matmul(rotation_matrix, extendedPoint)

	data.data = np.array(data.data)

	return data

def distance(p1, p2):
	# return np.sum(np.abs(p1 - p2))			# Manhattan distance 
	return np.sqrt(np.sum(np.pow(p1 - p2, 2))) 	# Euclidian distance

class Particle:
	def __init__(self, pos_mean, pos_std, vel_mean, vel_std):
		self.position = varied_point(pos_mean, pos_std)
		self.velocity = varied_point(vel_mean, vel_std)
	def force_vector(self, other_particle):
		disposition = self.position - other_particle.position
		return disposition / distance(self.position, other_particle.position)

def create_dataset_weak_clusters(output_file="weak_clusters.json", std=10, mean=[0, 0], clusters=10, points=1000, iterations=10, seed=42):
	random.seed(seed)

	np_mean = np.array(mean)

	nearest_force = points // clusters
	particles = []
	for p in range(points):
		particles.append(Particle(mean, std, mean, 0))

	for i in range(iterations):
		for p1 in particles:
			distance_pairs = []
			for p2 in particles:
				distance_pairs.append([p2, distance(p1.position, p2.position)])

			distance_pairs = sorted(distance_pairs, key=lambda x: x[1])

			c = 0
			for pair in distance_pairs:
				if (np.abs(np.sum(p1.position - pair[0].position) != 0)):
					if (c > nearest_force):
						p1.velocity += pair[0].force_vector(p1)
					else:
						p1.velocity -= pair[0].force_vector(p1) / (clusters - 1)

					c += 1


		for p in particles:
			# p.velocity /= nearest_force
			p.velocity *= 0.9
			p.position += p.velocity

		# print("Iteration #" + str(i + 1))
		# print([p.position for p in particles])
		# print([p.velocity for p in particles])
	data = Data([p.position for p in particles])
	data.save_data(output_file)

	return data

def pointFormatting(points):
	size = len(points[0])

	x_coords = [point[0] for point in points]
	z_coords = None

	if (size > 1):
		y_coords = [point[1] for point in points]
		if (size > 2):
			z_coords = [point[2] for point in points]
	else:
		y_coords = [0 for point in points]

	return (x_coords, y_coords, z_coords)

def plotPoints(points):
	plotPointSets([points])

def plotPointSets(sets):
	markers = ['o', 'v', '*']
	color = ['r', 'g', 'b']

	size = len(sets[0][0])

	fig = plt.figure()

	if (size == 3):
		ax = fig.add_subplot(111, projection='3d')
	else:
		ax = fig.add_subplot(111)


	for i, points in enumerate(sets):
		(x_coords, y_coords, z_coords) = pointFormatting(points)

		if (size == 3):
			ax.scatter(x_coords, y_coords, z_coords, c=color[i], marker=markers[i], label='Points')
		else:
			ax.scatter(x_coords, y_coords, c=color[i], marker=markers[i], label='Points')

	ax.legend()

	plt.show()

if __name__ == '__main__':
	print("Making line...")
	line_points = create_dataset_line(output_file="line.json", start=0, end=3, seed=time.time())

	print("Making square edge...")
	create_dataset_square_edge(output_file="square_edge.json", seed=time.time())

	print("Making square fill...")
	square_points = create_dataset_square_fill(output_file="square_fill.json", seed=time.time())

	print("Making eigth sphere fill...")
	create_dataset_eigth_sphere(output_file="eigth_sphere.json", seed=time.time())

	print("Making strong clusters...")
	create_dataset_strong_clusters(output_file="strong_clusters.json", seed=time.time())

	print("Line in 3D")
	rotate_into_dimention(line_points, seed=time.time()).save_data("3d_line.json")

	print("Square in 3D")
	rotate_into_dimention(square_points, seed=time.time()).save_data("3d_square.json")

	# print("Making weak clusters...")
	# create_dataset_weak_clusters(output_file="weak_clusters.json", seed=time.time())