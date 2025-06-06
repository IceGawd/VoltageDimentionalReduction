from voltage import *
from kmeans import *

def localSolver(data, landmarks, partitions, c, p_g):
	solvers = []
	voltages = []

	for i in range(len(landmarks)):
		landmark = landmarks[i]

		landmarkSolver = Solver(data)
		landmarkSolver.setWeights(gaussiankernel, c)
		landmarkSolver.addUniversalGround(p_g)
		landmarkSolver.addLandmark(landmark)
		landmarkVoltages = landmarkSolver.compute_voltages()

		voltages.append(landmarkVoltages)
		solvers.append(landmarkSolver)

		# landmarkSolver.plot(colored=True, name="../inputoutput/matplotfigures/" + name + "_voltages_" + str(i) + ".png")

	points = np.array(list(map(list, zip(*voltages))))

	pca = PCA(n_components=2)
	points_2d = pca.fit_transform(points)

	plt.scatter(points_2d[:, 0], points_2d[:, 1])
	plt.xlabel("PCA Component 1")
	plt.ylabel("PCA Component 2")
	plt.title("PCA Projection of Solver Outputs")
	plt.show()

	plt.savefig("../inputoutput/matplotfigures/" + name + "_PCA.png")

if __name__ == "__main__":
	name = "eigth_sphere"
	data = Data("../inputoutput/data/" + name + ".json", stream=False)
	# data = Data("square_fill.json", stream=False)

	# landmarks = [createLandmarkClosestTo(partitions.centers, [1, 0], 1), createLandmarkClosestTo(partitions.centers, [0, 1], 1)]
	landmarks = [
					createLandmarkClosestTo(data, [1, 0, 0], 1), 
					createLandmarkClosestTo(data, [0, 1, 0], 1), 
					createLandmarkClosestTo(data, [0, 0, 1], 1)
				]

	k = 100

	partitions = Partitions(data)
	partitions.k_means(k, seed=time.time())

	# c, p_g = bestParameterFinder(gaussiankernel, landmarks, partitions)
	# print(c, p_g)
	c = 0.03
	p_g = 0.00000006
	localSolver(data, landmarks, c, p_g)