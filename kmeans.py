from create_data import *

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

				print(point)
				print(self.centers[0])

				d = distance(point, self.centers[0])
				for center in self.centers:
					d = min(d, distance(point, center))

				distances.append(d)

			distances = np.array(distances)
			distances /= np.sum(distances)

			self.centers.append(weighted_random(self.data, distances))

		return self.centers

	def k_means(self, k, seed=42):
		if (seed != -1):
			random.seed(seed)
		
		self.centers = self.k_means_plus_plus(k)
		self.point_assignments = []										# This removes the benefit of streaming

		for c in range(k):
			self.point_assignments.append([])

		for i, point in enumerate(self.data):
			min_index = 0
			min_dist = distance(point, self.centers[0])
			
			for c in range(k - 1):
				dist = distance(point, self.centers[c + 1])
				if (min_dist > dist):
					min_index = c + 1
					min_dist = dist
			
			self.point_assignments[min_index].append([point, min_dist])

		updated_centers = []

		restart = False
		for points in self.point_assignments:
			if (len(points) != 0):
				center = np.zeros(len(self.data[0]))
				for pointPair in points:
					center += np.array(pointPair[0])

				center /= len(points)
				updated_centers.append(center)
			else:
				restart = True

		if restart:
			return self.k_means(k, -1)

		self.centers = updated_centers

		return self.centers, self.point_assignments

	def plot():
		plotPointSets([self.data, self.centers])

def approx_dimention(partitions, start=1, end=10, inc=1, seed=42):
	random.seed(seed)

	x = []
	y = []
	for k in range(start, end, inc):
		total_distance = 0
		for i, points in enumerate(partitions.point_assignments):
			for point in points:
				total_distance += distance(point[0], partitions.centers[i])

		x.append(np.log(k))
		y.append(np.log(total_distance / len(partitions.data)))

	coefficients = np.polyfit(np.array(x), np.array(y), 1)

	return -1 / coefficients[0]


if __name__ == '__main__':
	data = Data("large_line.json", stream=True)
	partitions = Partitions(data)

	partitions.k_means(10, seed=time.time())

	print("Approx Intrinsic Dimention: " + str(approx_dimention(partitions, seed=time.time())))