from create_data import *
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans

def weighted_random(values, weights):
	r = random.random()

	i = -1
	while r > 0:
		i += 1
		r -= weights[i]

	return values[i]

class Partitions():
	def __init__(self, data):
		self.data = data

	def k_means_plus_plus(self, k):
		# print(self.data.data)
		self.centers = [select_random(self.data)]

		for i in range(k - 1):
			distances = []

			for point in self.data:
				# print(type(point))
				# print(type(self.centers[0]))

				# print(point)
				# print(self.centers[0])

				d = distance(point, self.centers[0])
				for center in self.centers:
					d = min(d, distance(point, center))

				distances.append(d)

			distances = np.array(distances)
			distances /= np.sum(distances)

			self.centers.append(weighted_random(self.data, distances))

		return self.centers

	def k_means(self, k, seed=42, savePointAssignments=False):
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
				self.point_assignments[label].append([point, distance(point, self.centers[label])])

			# self.point_assignments = [data[kmeans.labels_ == i] for i in range(k)]	# k times less efficient
		self.voronoi = Voronoi(self.centers)

	def my_k_means(self, k, seed=42, savePointAssignments=False):
		if (seed != -1):
			random.seed(seed)
		
		self.centers = self.k_means_plus_plus(k)

		point_accumulator = [np.zeros(len(self.data[0])) for i in range(k)]
		point_counts = [0 for i in range(k)]

		if (savePointAssignments):														# This removes the benefit of streaming
			self.point_assignments = [[] for i in range(k)]

		for i, point in enumerate(self.data):
			min_index = 0
			min_dist = distance(point, self.centers[0])
			
			for c in range(k - 1):
				dist = distance(point, self.centers[c + 1])
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
		closest = []
		for i, point in enumerate(self.data):
			min_index = 0
			min_dist = distance(point, self.centers[0])
			
			for c in range(len(self.centers) - 1):
				dist = distance(point, self.centers[c + 1])
				if (min_dist > dist):
					min_index = c + 1
					min_dist = dist

			if (min_index == index):
				closest.append(i)

		return closest

	def plot(self, color='r', marker='o', ax=None, name=None):
		size = len(self.centers[0])

		if (ax == None):
			fig = plt.figure()

			if (size == 3):
				ax = fig.add_subplot(111, projection='3d')
			else:
				ax = fig.add_subplot(111)

		if (size == 3):
			(x_coords, y_coords, z_coords) = pointFormatting(self.centers)
			ax.scatter(x_coords, y_coords, z_coords, c=color, marker=marker, label='Centers')
		else:
			(x_coords, y_coords, z_coords) = pointFormatting(self.data)
			ax.scatter(x_coords, y_coords, c=color, marker=marker, label='Points')

			voronoi_plot_2d(self.voronoi, ax=ax, show_vertices=False, line_colors='blue', line_width=1, line_alpha=0.6)

		ax.legend()

		if (name):
			plt.savefig(name)

		plt.show()
		# plotPointSets([self.data, self.centers])

def approx_dimention(data, start=1, end=10, inc=1, seed=42):
	random.seed(seed)

	x = []
	y = []

	for k in range(start, end, inc):
		partitions = Partitions(data)
		partitions.k_means(k, savePointAssignments=True, seed=-1)

		total_distance = 0
		for i, points in enumerate(partitions.point_assignments):
			for point in points:
				total_distance += distance(point[0], partitions.centers[i])

		x.append(np.log(k))
		y.append(np.log(total_distance / len(partitions.data)))

	coefficients = np.polyfit(np.array(x), np.array(y), 1)

	return -1 / coefficients[0]


if __name__ == '__main__':
	data = Data("strong_clusters.json", stream=False)
	partitions = Partitions(data)

	partitions.k_means(10, seed=time.time())
	partitions.plot()

	print("Approx Intrinsic Dimention: " + str(approx_dimention(data, seed=time.time())))