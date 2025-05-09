import create_data
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans
import time
import numpy as np
import matplotlib.pyplot as plt
import random

def weighted_random(values, weights):
	r = random.random()

	i = -1
	while r > 0:
		i += 1
		r -= weights[i]

	return values[i]

class DistanceBased():
	def __init__(self):
		self.setDistanceEuclidean()

	def setDistanceEuclidean(self):
		self.distance = lambda x, y: np.linalg.norm(x - y)

	def setDistanceManhattan(self):
		self.distance = lambda x, y: np.sum(np.abs(x - y))

	def setDistanceLInfinity(self):
		self.distance = lambda x, y: np.max(np.abs(x - y))

class Partitions(DistanceBased):
	"""Using K-means to partition a large dataset"""
	def __init__(self, data):
		self.data = data
		super().__init__()

	def k_means_plus_plus(self, k):
		"""The old k-means++ algorithm before using sci-kit"""

		# print(self.data.data)
		self.centers = [create_data.select_random(self.data)]

		for i in range(k - 1):
			distances = []

			for point in self.data:
				# print(type(point))
				# print(type(self.centers[0]))

				# print(point)
				# print(self.centers[0])

				d = self.distance(point, self.centers[0])
				for center in self.centers:
					d = min(d, self.distance(point, center))

				distances.append(d)

			distances = np.array(distances)
			distances /= np.sum(distances)

			self.centers.append(weighted_random(self.data, distances))

		return self.centers

	def k_means(self, k, seed=42, savePointAssignments=False):
		"""Runs k-means and saves the centers and point counts. With option to save pointAssignments for voronoi drawing"""
		if (seed == -1):
			kmeans = KMeans(n_clusters=k, init="k-means++").fit(self.data)
		else:
			kmeans = KMeans(n_clusters=k, random_state=int(seed), init="k-means++", n_init=1).fit(self.data)

		self.k = k
		self.centers = kmeans.cluster_centers_
		self.point_counts = np.bincount(kmeans.labels_).tolist()

		if savePointAssignments:
			self.point_assignments = [[] for i in range(k)]
			for i, point in enumerate(data):
				label = kmeans.labels_[i]

				# print(point)
				# print(self.centers[label])
				# print(self.distance(point, self.centers[label]))
				self.point_assignments[label].append([point, self.distance(point, self.centers[label])])

			# self.point_assignments = [data[kmeans.labels_ == i] for i in range(k)]	# k times less efficient
		# self.voronoi = Voronoi(self.centers)

	def my_k_means(self, k, seed=42, savePointAssignments=False):
		"""The old k-means algorithm"""

		if (seed != -1):
			random.seed(seed)
		
		self.centers = self.k_means_plus_plus(k)

		point_accumulator = [np.zeros(len(self.data[0])) for i in range(k)]
		point_counts = [0 for i in range(k)]

		if (savePointAssignments):														# This removes the benefit of streaming
			self.point_assignments = [[] for i in range(k)]

		for i, point in enumerate(self.data):
			min_index = 0
			min_dist = self.distance(point, self.centers[0])
			
			for c in range(k - 1):
				dist = self.distance(point, self.centers[c + 1])
				if (min_dist > dist):
					min_index = c + 1
					min_dist = dist
			
			if (savePointAssignments):
				self.point_assignments[min_index].append([point, min_dist])

			point_accumulator[min_index] += point
			point_counts[min_index] += 1

		updated_centers = []
		self.point_counts = []

		for acc, count in zip(point_accumulator, point_counts):
			if (count != 0):
				updated_centers.append(acc / count)
				self.point_counts.append(count)

		self.centers = updated_centers
		self.voronoi = Voronoi(self.centers)

	def getClosestPoints(self, index):
		"""
		Finds the points whose closest points are the point indicated by the index

		Args:
			index (int): the index of the point

		Returns:
			List[np.ndarray]: All the points whose closest point is data[index]

		"""
		closest = []
		for i, point in enumerate(self.data):
			min_index = 0
			min_dist = self.distance(point, self.centers[0])
			
			for c in range(len(self.centers) - 1):
				dist = self.distance(point, self.centers[c + 1])
				if (min_dist > dist):
					min_index = c + 1
					min_dist = dist

			if (min_index == index):
				closest.append(i)

		return closest

	def plot(self, color='r', marker='o', ax=None, name=None):
		"""Plot the kmeans"""
		plot = create_data.Plotter()

		size = len(self.centers[0])

		if (ax == None):
			fig = plt.figure()

			if (size == 3):
				ax = fig.add_subplot(111, projection='3d')
			else:
				ax = fig.add_subplot(111)

		if (size == 3):
			(x_coords, y_coords, z_coords) = plot.pointFormatting(self.centers)
			ax.scatter(x_coords, y_coords, z_coords, c=color, marker=marker, label='Centers')
		else:
			(x_coords, y_coords, z_coords) = plot.pointFormatting(self.data)
			ax.scatter(x_coords, y_coords, c=color, marker=marker, label='Points')

			# voronoi_plot_2d(self.voronoi, ax=ax, show_vertices=False, line_colors='blue', line_width=1, line_alpha=0.6)

		ax.legend()

		if (name):
			plt.savefig(name)

		plt.show()
		# plotPointSets([self.data, self.centers])

class DimentionApproximator(DistanceBased):
	def __init__(self, data):
		self.data = data
		super().__init__()

	def approx_dimention(self, start=1, end=10, inc=1, seed=42):
		random.seed(seed)

		x = []
		y = []

		for k in range(start, end, inc):
			partitions = Partitions(self.data)
			partitions.k_means(k, savePointAssignments=True, seed=-1)

			total_distance = 0
			for i, points in enumerate(partitions.point_assignments):
				for point in points:
					total_distance += self.distance(point[0], partitions.centers[i])

			x.append(np.log(k))
			y.append(np.log(total_distance / len(partitions.data)))

		coefficients = np.polyfit(np.array(x), np.array(y), 1)

		return -1 / coefficients[0]


if __name__ == '__main__':
	data = create_data.Data("../inputoutput/data/spiral.json", stream=True)
	partitions = Partitions(data)

	partitions.k_means(10, seed=time.time())
	# partitions.plot()

	approxer = DimentionApproximator(data)

	print("Approx Intrinsic Dimention: " + str(approxer.approx_dimention(seed=time.time())))