import numpy as np
import json
import ijson
import pickle
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
import time
import types
import tempfile
import os
os.environ["PYTHONUNBUFFERED"] = "1"

def select_random(array):
	"""Selects a random element from an array."""
	return array[int(len(array) * random.random())]

def pointFormatting(points):
	"""Formats points into coordinate lists for plotting."""
	size = len(points[0])
	x_coords = [point[0] for point in points]
	z_coords = None
	if size > 1:
		y_coords = [point[1] for point in points]
		if size > 2:
			z_coords = [point[2] for point in points]
	else:
		y_coords = [0 for point in points]
	return (x_coords, y_coords, z_coords)

def plotPoints(points, name=None):
	"""Plots a set of points in 2D or 3D."""
	plotPointSets([points], name)

def plotPointSets(sets, name=None):
	"""Plots multiple sets of points with different colors and markers."""
	markers = ['o', 'v', '*']
	color = ['r', 'g', 'b']
	size = len(sets[0][0])
	fig = plt.figure()
	if size == 3:
		ax = fig.add_subplot(111, projection='3d')
	else:
		ax = fig.add_subplot(111)
	for i, points in enumerate(sets):
		(x_coords, y_coords, z_coords) = pointFormatting(points)
		if size == 3:
			ax.scatter(x_coords, y_coords, z_coords, c=color[i], marker=markers[i], label='Points')
		else:
			ax.scatter(x_coords, y_coords, c=color[i], marker=markers[i], label='Points')
	ax.legend()
	if name:
		plt.savefig(name)
	plt.show()

def stream_save(output_file, data_generator, *args):
	"""Saves data to a JSON file in a streaming manner."""
	with open(output_file, "w") as f:
		f.write("{\"data\": [\n")
		first = True
		length = 0
		for array in data_generator(*args):
			if not first:
				f.write(", \n")
			json.dump(list(array), f)
			length += 1
			first = False
		f.write("], \n\"length\": " + str(length) + "}")

def linear_generator(data):
	"""Yields data points one by one."""
	for d in data.tolist():
		yield d

class Data():
	"""Class for handling and processing data sets."""
	def __init__(self, arg=None, stream=False):
		self.stream = stream

		if isinstance(arg, list):
			self.data = np.array(arg)
			self.length = len(self.data)
		elif isinstance(arg, str):
			if (stream):
				self.data = self.stream_data_json(arg)
				self.length = next(self.data)
				self.i = 0
			else:
				self.load_data(arg)
				self.length = len(self.data)

			self.input_file = arg
		else:
			self.data = arg
			self.length = len(self.data)

	# Len can run on Data
	def __len__(self):
		return self.length

	# Data can be get indexed
	def __getitem__(self, index):
		if (self.stream):
			if (index < self.i):
				self.data = self.stream_data_json(self.input_file)
				next(self.data)
				self.i = 0

			while (self.i <= index):
				value = next(self.data)
				self.i += 1

			return value
		else:
			return self.data[index]

	# """
	def __setitem__(self, index, value):
		self.data[index] = value
	# """

	# Make Data able to be for looped
	def __iter__(self):
		if (hasattr(self, 'input_file')):
			self.streaming_data = self.stream_data_json(self.input_file)
			next(self.streaming_data)
		else:
			self.streaming_data = 0

		return self

	def __next__(self):
		try:
			if (hasattr(self, 'input_file')):
				return np.array(next(self.streaming_data))
			else:
				if (self.streaming_data == self.length):
					raise
				else:
					return np.array(self.data[self.streaming_data])

				self.streaming_data += 1
		except StopIteration:
			raise

	def getSubSet(self, indexList):
		"""Returns a subset of the data given a list of indices."""
		subset = []
		for index in indexList:
			subset.append(self.data[index])
		return Data(subset)

	def save_data_json(self, output_file):
		stream_save(output_file, linear_generator, self.data)

	def save_data_pickle(self, output_file):
		with open(output_file, 'wb') as f: 
			pickle.dump(self.data, f) 

	def load_data_json(self, input_file):
		with open(input_file, 'r') as f:
			self.input_file = input_file

			data = json.load(f)
			self.data = data["data"]
			self.length = data["length"]
			for i, point in enumerate(self.data):
				self.data[i] = np.array(point)

			return self.data

	def load_data_pickle(self, input_file):
		with open(input_file, 'r') as f:
			self.input_file = input_file
			self.data = pickle.load(f)

			return self.data

	def stream_data_json(self, input_file):
		"""Stream the dataset if its saved in a json file"""
		with open(input_file, 'rb') as f:
			f.seek(0, 2)
			position = f.tell()

			value = ""
			read = False
			while position > 0:
				position -= 1
				f.seek(position)
				byte = f.read(1)

				if byte == b' ':
					# print(value)
					yield int(value)
					break

				if (read):
					value = byte.decode() + value

				if byte == b'}':
					read = True

		with open(input_file, 'r') as f:
			f.readline()

			for line in f:
				if ("length" in line):
					break

				data = json.loads(line.strip().split(']')[0] + ']')
				yield np.array(data)

	file_function_pairs = [["json", save_data_json, load_data_json], ["pkl", save_data_pickle, load_data_pickle]]

	def data_function(self, file, save_or_load):
		"""Used for saving and loading the dataset"""
		if (file == None):
			return

		for ffp in self.file_function_pairs:
			if file[-len(ffp[0]):] == ffp[0]:
				if save_or_load == 1:
					ffp[save_or_load](self.data, file)
				else:
					return ffp[save_or_load](self, file)

	def save_data(self, output_file):
		self.data_function(output_file, 1)

	def load_data(self, input_file):
		self.data_function(input_file, 2)

	def get_random_point(self):
		"""Returns a random point from the dataset."""
		return select_random(self.data)

	def plot(self, name=None):
		"""Plots the dataset."""
		plotPoints(self.data, name)


def stream_dataset_creator(output_file, function, seed, stream, *args):
	"""Creates a dataset by passing in generator functions, allowing for streamed and not streamed dataset creation"""
	random.seed(seed)

	if (stream):
		stream_save(output_file, function, *args)
		data = Data(output_file, stream=True)
	else:
		data = []

		for point in function(*args):
			data.append(point)

		data = Data(data)
		data.save_data(output_file)

	return data

def line_generator(start, end, points):
	"""Generates points along a line in 1D space."""
	for _ in range(points):
		yield np.array([random.random() * (end - start) + start])

def create_dataset_line(output_file=None, start=0, end=1, points=1000, seed=42, stream=False):
	"""Generates a dataset of a 1D line"""
	return stream_dataset_creator(output_file, line_generator, seed, stream, start, end, points)

def create_dataset_square_edge(output_file=None, p1=(0,0), p2=(1,1), points=1000, seed=42):
	"""Generates a dataset of the edge of a square"""
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

def create_dataset_square_fill(output_file=None, p1=(0,0), p2=(1,1), points=1000, seed=42):
	"""Generates a dataset of a filled in square"""
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

def eigth_sphere_generator(radius, x_pos, y_pos, z_pos, points):
	"""Generator for points on an eigth sphere"""
	for p in range(points):
		z = random.random()						# Z value
		angleXY = np.pi * random.random() / 2	# Angle in the XY plane

		yield np.array([radius * np.sqrt(1 - z**2) * np.cos(angleXY) * (2 * x_pos - 1), radius * np.sqrt(1 - z**2) * np.sin(angleXY) * (2 * y_pos - 1), radius * z * (2 * z_pos - 1)])


def create_dataset_eigth_sphere(output_file=None, radius=1, x_pos=True, y_pos=True, z_pos=True, points=1000, seed=42, stream=False):
	"""Generates a dataset of an eigth sphere"""
	return stream_dataset_creator(output_file, eigth_sphere_generator, seed, stream, radius, x_pos, y_pos, z_pos, points)

def dimentional_variation(dimentions):
	"""Returns an np array that is full of random variables from -inf to inf based on the standard normal distribution"""
	z_vals = []
	for d in range(dimentions):
		z_vals.append(stats.norm.ppf(random.random()))

	return np.array(z_vals)

def varied_point(mean, std):
	return mean + std * dimentional_variation(len(mean))

def strong_cluster_generator(internal_std, cluster_centers, points):
	"""Generates points in a strong cluster"""
	c = -1
	for p in range(points):
		if (p / points >= c / 100):
			c += 1
			# print(str(c) + "%")

		yield varied_point(select_random(cluster_centers), internal_std)


def create_dataset_strong_clusters(output_file=None, internal_std=1, external_std=10, mean=[0, 0], clusters=10, points=1000, seed=42, stream=False):
	"""Generates a strongly clustered datapoint by selecting cluster centers and variance in the clusters via normal distribution"""
	data = []
	random.seed(seed)

	np_mean = np.array(mean)

	cluster_centers = []
	for c in range(clusters):
		cluster_centers.append(varied_point(np_mean, external_std))

	if (stream):
		stream_save(output_file, strong_cluster_generator, internal_std, cluster_centers, points)
		data = Data(output_file, stream=True)
	else:
		for p in strong_cluster_generator(internal_std, cluster_centers, points):
			data.append(p)

		data = Data(data)
		data.save_data(output_file)

	return data

def rotate_into_dimention(data, higher_dim=3, seed=42):
	"""Moves the data into a higher dimention and does rotations centered at the origin"""
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

	# print(rotation_matrix)

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
	"""NOT USED: represents a weakly clustered datapoint"""
	def __init__(self, pos_mean, pos_std, vel_mean, vel_std):
		self.position = varied_point(pos_mean, pos_std)
		self.velocity = varied_point(vel_mean, vel_std)
	def force_vector(self, other_particle):
		disposition = self.position - other_particle.position
		return disposition / distance(self.position, other_particle.position)

def create_dataset_weak_clusters(output_file=None, std=10, mean=[0, 0], clusters=10, points=1000, iterations=10, seed=42):
	"""NOT USED: makes weakly clustered data"""
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

def spiral_generator(radius, center, rotations, height, points):
	line = 2 * np.pi * rotations
	heightPerRadian = height / line

	for p in range(points):
		d = random.random() * line
		yield np.array([np.cos(d), np.sin(d), heightPerRadian * d])

def create_dataset_spiral(output_file=None, radius=1, center=[0, 0], rotations=3, height=10, points=1000, seed=42, stream=False):
	return stream_dataset_creator(output_file, spiral_generator, seed, stream, radius, center, rotations, height, points)

if __name__ == '__main__':
	print("Making line...")
	line_points = create_dataset_line(output_file="../inputoutput/data/line.json", start=0, end=3, seed=time.time())

	print("Making square edge...")
	edge_points = create_dataset_square_edge(output_file="../inputoutput/data/square_edge.json", seed=time.time())

	print("Making square fill...")
	square_points = create_dataset_square_fill(output_file="../inputoutput/data/square_fill.json", seed=time.time())

	print("Making eigth sphere fill...")
	create_dataset_eigth_sphere(output_file="../inputoutput/data/eigth_sphere.json", seed=time.time())

	print("Making strong clusters...")
	create_dataset_strong_clusters(output_file="../inputoutput/data/strong_clusters.json", seed=time.time())

	print("Making spiral...")
	create_dataset_spiral(output_file="../inputoutput/data/spiral.json", seed=time.time())

	print("Line in 3D")
	rotate_into_dimention(line_points, seed=time.time()).save_data("../inputoutput/data/3d_line.json")

	print("Square Fill in 3D")
	rotate_into_dimention(square_points, seed=time.time()).save_data("../inputoutput/data/3d_square.json")

	print("Square Edge in 10D")
	rotate_into_dimention(edge_points, higher_dim=10, seed=time.time()).save_data("../inputoutput/data/10d_square_edge.json")

	"""
	print("Large Line")
	create_dataset_line(output_file="../inputoutput/data/large_line.json", points=1000000, seed=time.time(), stream=True)

	print("Large Single Cluster")
	create_dataset_strong_clusters(output_file="../inputoutput/data/large_single_cluster.json", clusters=1, points=1000000, seed=time.time(), stream=True)

	print("Large Many Clusters")
	create_dataset_strong_clusters(output_file="../inputoutput/data/large_many_clusters.json", internal_std=100, external_std=1000, mean=[0, 0, 0], clusters=10000, points=1000000, seed=time.time(), stream=True)

	print("Large eigth sphere")
	create_dataset_eigth_sphere(output_file="../inputoutput/data/large_eigth_sphere.json", points=1000000, seed=time.time(), stream=True)

	print("Large spiral")
	create_dataset_spiral(output_file="../inputoutput/data/large_spiral.json", points=1000000, seed=time.time(), stream=True)
	# """

	# data = Data("large_line.json", stream=True)
	# for point in data:
	# 	print(point)

	# print("Making weak clusters...")
	# create_dataset_weak_clusters(output_file="weak_clusters.json", seed=time.time())