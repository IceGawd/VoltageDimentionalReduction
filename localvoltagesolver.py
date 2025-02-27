from voltage import *
from kmeans import *
from sklearn.decomposition import PCA

# Example usage
if __name__ == "__main__":
	name = "eigth_sphere"
	data = Data(name + ".json", stream=False)
	# data = Data("square_fill.json", stream=False)

	k = 100

	partitions = Partitions(data)
	partitions.k_means(k, seed=time.time())

	# c, p_g = bestParameterFinder(gaussiankernel, landmarks, partitions)
	# print(c, p_g)
	c = 0.03
	p_g = 0.00000006

	landmarks = [
					createLandmarkClosestTo(partitions.centers, [1, 0, 0], 1), 
					createLandmarkClosestTo(partitions.centers, [0, 1, 0], 1), 
					createLandmarkClosestTo(partitions.centers, [0, 0, 1], 1)
				]

	# landmarks = [createLandmarkClosestTo(partitions.centers, [1, 0], 1), createLandmarkClosestTo(partitions.centers, [0, 1], 1)]
	solvers = []
	voltages = []

	for i in range(len(landmarks)):
		landmark = landmarks[i]

		landmarkSolver = Solver(partitions.centers)
		landmarkSolver.setPartitionWeights(gaussiankernel, partitions, c)
		landmarkSolver.addUniversalGround(p_g)
		landmarkSolver.addLandmark(landmark)
		landmarkVoltages = landmarkSolver.compute_voltages()

		solvers.append(landmarkSolver)

		# landmarkSolver.plot(colored=True, name=name + "_voltages_" + str(i) + ".png")

	points = np.array(list(map(list, zip(*(solver.localSolver(data, partitions, c) for solver in solvers)))))

	pca = PCA(n_components=2)
	points_2d = pca.fit_transform(points_array)

	plt.scatter(points_2d[:, 0], points_2d[:, 1])
	plt.xlabel("PCA Component 1")
	plt.ylabel("PCA Component 2")
	plt.title("PCA Projection of Solver Outputs")
	plt.show()

	plt.savefig(name + "_PCA.png")