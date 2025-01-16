import create_data

def distance(p1, p2):
	return np.sum(np.abs(p1 - p2))					# Manhattan distance 
	# return np.sqrt(np.sum(np.pow(p1 - p2, 2))) 	# Euclidian distance

def weighted_random(values, weights):
	r = random.random()

	i = 0
	while r > 0:
		r -= weights[i]

	return values[i]

def k_means_plus_plus(data, k):
	centers = [select_random(data)]

	for i in range(k - 1):
		distances = []

		for point in data:
			distance = distance(point, centers[0])
			for center in centers:
				distance = min(distance, distance(point, center))

			distances.append(distance)

		distances = np.array(distances)
		distances /= np.sum(distances)

		centers.append(weighted_random(points, distances))

	return centers

def k_means(data, k, seed=42):
	random.seed(seed)
	
	centers = k_means_plus_plus(data, k)

	

if __name__ == '__main__':
	pass