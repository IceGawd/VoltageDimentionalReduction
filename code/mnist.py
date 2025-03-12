import create_data
import voltage
import kmeans
import localvoltagesolver

from sklearn.datasets import fetch_openml
import numpy as np

if __name__ == "__main__":
	print("Loading Data...")
	mnist = fetch_openml('mnist_784', version=1, as_frame=False)
	X, y = mnist.data, mnist.target.astype(np.int64)

	# data = create_data.Data(np.array(X))

	landmarks = []
	subDivision = {}
	summation = {}
	count = {}

	print("Sorting and averaging...")

	for xi, yi in zip(X, y):
		if yi in summation:
			subDivision[yi].append(np.array(xi))
			summation[yi] += np.array(xi)
			count[y1] += 1
		else:
			subDivision[yi] = [np.array(xi)]
			summation[yi] = np.array(xi)
			count[y1] = 1

	print("Kmeans...")

	k = 100
	data = []

	for yi in range(10):
		partitions = kmeans.Partitions(subDivision[yi])
		partitions.k_means(k // 10, seed=time.time())

		landmarks.append(voltage.createLandmarkClosestTo(partitions.centers, summation[yi] / count[yi], 1))
		data += partitions.centers

	data = create_data.Data(data)

	partitions = kmeans.Partitions(data)
	partitions.k_means(k, seed=time.time())

	print("Parameter Finding...")

	c, p_g = voltage.bestParameterFinder(voltage.gaussiankernel, landmarks, partitions, emin=-10, emax=-1, mantissa=True, L=0)
	print(c, p_g)
	# localvoltagesolver.localSolver(data, landmarks, c, p_g)