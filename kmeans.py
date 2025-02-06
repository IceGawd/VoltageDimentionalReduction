from create_data import *

def weighted_random(values, weights):
	r = random.random()

	i = -1
	while r > 0:
		i += 1
		r -= weights[i]

	return values[i]

class Partitions(object):
	def __init__(self, data):
		self.data = data

	def k_means_plus_plus(self, k):
		self.centers = [select_random(data)]

		for i in range(k - 1):
			distances = []

			for point in data:
				d = distance(point, centers[0])
				for center in self.centers:
					d = min(d, distance(point, center))

				distances.append(d)

			distances = np.array(distances)
			distances /= np.sum(distances)

			self.centers.append(weighted_random(data, distances))

		return self.centers

	def k_means(data, k, seed=42):
		if (seed != -1):
			random.seed(seed)
		
		self.centers = k_means_plus_plus(data, k)
		point_assignments = []

		for c in range(k):
			point_assignments.append([])

		for i, point in enumerate(data):
			min_index = 0
			min_dist = distance(point, centers[0])
			
			for c in range(k - 1):
				dist = distance(point, centers[c + 1])
				if (min_dist > dist):
					min_index = c + 1
					min_dist = dist
			
			point_assignments[min_index].append([point, min_dist])

		updated_centers = []

		restart = False
		for points in point_assignments:
			if (len(points) != 0):
				center = np.zeros(len(data[0]))
				for pointPair in points:
					center += np.array(pointPair[0])

				center /= len(points)
				updated_centers.append(center)
			else:
				restart = True

		if restart:
			return k_means(data, k, -1)

		return updated_centers, point_assignments

def approx_dimention(data, start=1, end=10, inc=1, seed=42):
	random.seed(seed)

	x = []
	y = []
	for k in range(start, end, inc):
		centers, point_assignments = k_means(data, k, -1)

		total_distance = 0
		for i, points in enumerate(point_assignments):
			for point in points:
				total_distance += distance(point[0], centers[i])

		x.append(np.log(k))
		y.append(np.log(total_distance / len(data)))

	coefficients = np.polyfit(np.array(x), np.array(y), 1)

	return -1 / coefficients[0]



if __name__ == '__main__':
	data = load_data_json("3d_square.json")
	centers, pa = k_means(data, 10, seed=time.time())
	plotPointSets([data, centers])
	print("Approx Intrinsic Dimention: " + str(approx_dimention(data, seed=time.time())))