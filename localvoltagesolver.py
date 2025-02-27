from voltage import *
from kmeans import *

# Example usage
if __name__ == "__main__":
	data = Data("eigth_sphere.json", stream=False)
	# data = Data("square_fill.json", stream=False)

	k = 100

	partitions = Partitions(data)
	partitions.k_means(k, seed=time.time())

	# c, p_g = bestParameterFinder(gaussiankernel, landmarks, partitions)
	# print(c, p_g)
	c = 0.1
	p_g = 0.001

	landmarks = [createLandmarkClosestTo(data, [1, 0, 0], 1), createLandmarkClosestTo(data, [0, 1, 0], 1), createLandmarkClosestTo(data, [0, 0, 1], 1)]
	# landmarks = [createLandmarkClosestTo(partitions.centers, [1, 0], 1), createLandmarkClosestTo(partitions.centers, [0, 1], 1)]
	solvers = []

	for i in range(len(landmarks)):
		landmark = landmarks[i]

		landmarkSolver = Solver(partitions.centers)
		landmarkSolver.setPartitionWeights(gaussiankernel, partitions, c)
		landmarkSolver.addUniversalGround(p_g)
		landmarkSolver.addLandmark(landmark)
		landmarkVoltages = landmarkSolver.compute_voltages()

		solvers.append(landmarkSolver)
		landmarkSolver.plot(colored=True, name="eigth_sphere_voltages_" + str(i) + ".png")

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
		localSolver.setWeights(gaussiankernel, c)
		localSolver.addLandmarks(closeLandmarks)
		localVoltages = localSolver.compute_voltages()

		for i, v in zip(closestIndicies, localVoltages):
			voltages[i] = v

	temp = Solver(data)
	temp.voltages = voltages
	temp.plot()