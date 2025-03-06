import create_data
import voltage
import kmeans
import localvoltagesolver

if __name__ == "__main__":
	mnist = fetch_openml('mnist_784', version=1, as_frame=False)
	X, y = mnist.data, mnist.target.astype(np.int64)

	data = create_data.Data(np.array(X))

	landmarks = []
	summation = {}
	count = {}

	for xi, yi in zip(X, y):
		if yi in summation:
			summation[yi] += np.array(xi)
			count[y1] += 1
		else:
			summation[yi] = np.array(xi)
			count[y1] = 1

	for yi in range(10):
		landmarks.append(voltage.createLandmarkClosestTo(mnist_data, summation[yi] / count[yi], 1))

	k = 100

	partitions = kmeans.Partitions(data)
	partitions.k_means(k, seed=time.time())

	c, p_g = bestParameterFinder(voltage.gaussiankernel, landmarks, partitions)
	print(c, p_g)
	localvoltagesolver.localSolver(data, landmarks, c, p_g)
