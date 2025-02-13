from voltage import *
from kmeans import *

# Example usage
if __name__ == "__main__":
	data = Data("strong_clusters.json", stream=False)

	partitions = Partitions(data)
	partitions.k_means(10, seed=time.time())

	most_central_index = 0

	for index in range(len(partitions.centers)):
		if distance(partitions.centers[index], [0, 0]) > distance(partitions.centers[most_central_index], [0, 0]):
			most_central_index = index

	# print(partitions.centers[most_central_index])

	landmarks = [Landmark(index, 1)]
	c = bestCFinder(gaussiankernel, landmarks, partitions)

	meanSolver = Solver(partitions.centers)
	meanSolver.setWeights(kernel=gaussiankernel, c=c, partition=partitions)
	meanSolver.addUniversalGround()
	meanSolver.addLandmarks(landmarks)
	meanSolver.compute_voltages()
	meanSolver.plot()