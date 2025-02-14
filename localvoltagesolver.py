from voltage import *
from kmeans import *

# Example usage
if __name__ == "__main__":
	data = Data("strong_clusters.json", stream=False)

	k = 10

	partitions = Partitions(data)
	partitions.k_means(k, seed=time.time())

	most_central_index = 0

	for index in range(len(partitions.centers)):
		if distance(partitions.centers[index], [0, 0]) > distance(partitions.centers[most_central_index], [0, 0]):
			most_central_index = index

	# print(partitions.centers[most_central_index])

	landmarks = [Landmark(index, 1)]
	c = bestCFinder(gaussiankernel, landmarks, partitions)

	landmarkSolver = Solver(partitions.centers)
	landmarkSolver.setPartitionWeights(kernel=gaussiankernel, c=c, partition=partitions)
	landmarkSolver.addUniversalGround()
	landmarkSolver.addLandmarks(landmarks)
	landmarkVoltages = landmarkSolver.compute_voltages()

	voltages = [0 for i in range(len(data))]

	for index in range(k):
		closestIndicies = partitions.getClosestPoints(index)
		closeLandmarksIndicies = []

		for pair in partitions.voronoi.ridge_points:
			if pair[0] == index:
				closeLandmarksIndicies.append(pair[1])
			if pair[1] == index:
				closeLandmarksIndicies.append(pair[0])

		closeLandmarks = []
		for cli in closeLandmarksIndicies:
			closeLandmarks.append(Landmark(cli, landmarkVoltages[cli]))

		localSolver = Solver(data.getSubSet(closestIndicies))
		localSolver.setWeights(kernel=gaussiankernel, c=c)
		localSolver.addLandmarks(closeLandmarks)
		localVoltages = localSolver.compute_voltages()

		for i, v in zip(closestIndicies, localVoltages):
			voltages[i] = v

	temp = Solver(data)
	temp.voltages = voltages
	temp.plot()