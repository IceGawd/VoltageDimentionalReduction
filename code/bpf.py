import voltage
import kmeans
import create_data
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

dist = []


def nInfUniform(voltages):
	voltages.sort()
	uniform = np.array([x / (len(voltages) - 1) for x in range(len(voltages))])

	return np.linalg.norm(abs(voltages - uniform))

def nInfExp(voltages, base=10):
	global dist

	voltages.sort()

	if (len(dist) != len(voltages)):
		dist = np.array([np.pow(base, (x / (len(voltages) - 1)) - 1) for x in range(len(voltages))])

	return np.linalg.norm(abs(voltages - dist))

def median(voltages, value=0.5):
	voltages.sort()
	return abs(voltages[int(len(voltages) / 2)] - value)

def minimum(voltages, value=0.1):
	voltages.sort()
	return abs(voltages[0] - value)

def minWithStd(voltages, value=0.1):
	voltages.sort()
	return abs(voltages[0] - value) / np.std(voltages)

def calculateFor(kernel, landmarks, partition, metric, c, p_g):
	meanSolver = voltage.Solver(partition.centers)
	meanSolver.setPartitionWeights(kernel, partition, np.exp(c))
	meanSolver.addUniversalGround(np.exp(p_g))
	meanSolver.addLandmarks(landmarks)

	voltages = np.array(meanSolver.compute_voltages())

	if (metric):
		return metric(voltages)
	else:
		return voltages, meanSolver

def bestParameterFinder(kernel, landmarks, partition, metric=nInfUniform, minBound=-25, maxBound=-1, granularity=5, epsilon=1):
	"""
	Finds the best parameters (C and P_G) for a solver based on voltage distribution minimization.

	This function searches for optimal parameters `C` and `P_G` by iterating over exponent values in 
	a specified range, computing voltages using a solver, and minimizing some metric
	between the voltage distribution and a uniform distribution.

	Parameters:
	-----------
	kernel : object
		The kernel function or object used to compute partition weights.
	landmarks : list
		A list of landmark points used in the solver.
	partition : object
		A partition object containing centers used in the solver.
	nInfUniform : function (list of floating point values -> floating point value)
		A function that is used to quantify if voltages are good or bad, the smaller the better
	minBound : int, optional (default=1e-5)
		The minimum value to consider for `C` and `P_g` as 10^minBound.
	maxBound : int, optional (default=1e5)
		The maximum value to consider for `C` and `P_g` as 10^maxBound.

	Returns:
	--------
	tuple
		A tuple (bestC, bestG), where:
		- bestC (float): The optimized value for parameter C.
		- bestG (float): The optimized value for parameter P_g.
	"""

	window_size = (maxBound - minBound) / 2

	bestc = minBound + window_size
	bestg = minBound + window_size

	val = float('inf')

	while window_size > epsilon:
		print(window_size, np.exp(bestc), np.exp(bestg))

		cs = [bestc + x * window_size / granularity for x in range(-granularity + 1, granularity)]
		gs = [bestg + x * window_size / granularity for x in range(-granularity + 1, granularity)]

		for c in cs:
			for g in gs:
				tempval = calculateFor(kernel, landmarks, partition, metric, c, g)

				# print(tempval)
				if (val > tempval):
					# print(c)
					# print(g)
					# print(tempval)

					bestc = c
					bestg = g
					val = tempval

		window_size /= granularity

	return bestc, bestg

if __name__ == "__main__":
	print("Loading data...")
	triangle = create_data.Data("../inputoutput/data/large_triangle.json", stream=True)

	print("Doing partitioning...")
	k = 200

	partitions = kmeans.Partitions(triangle)
	partitions.k_means(k, seed=time.time())

	landmarks = [voltage.createLandmarkClosestTo(partitions.centers, [0, 0], 1)]
	allLandmarks = [voltage.createLandmarkClosestTo(partitions.centers, [1, 1], 1), voltage.createLandmarkClosestTo(partitions.centers, [2, 0], 1)]

	for metric in [nInfUniform, nInfExp, median, minimum, minWithStd]:
		for kernel in [voltage.gaussiankernel]:
			c, p_g = bestParameterFinder(kernel, landmarks, partitions, metric=metric)
			print("Metric: " + metric.__name__ + " Kernel: " + kernel.__name__ + " c: " + str(np.exp(c)) + " p_g: " + str(np.exp(p_g)))

			singlevoltage, meanSolver = calculateFor(kernel, landmarks, partitions, None, c, p_g)
			# print(singlevoltage)

			voltages = [singlevoltage]
			for landmark in allLandmarks:
				voltages.append(calculateFor(kernel, [landmark], partitions, None, c, p_g)[0])

			points = np.array(list(map(list, zip(*voltages))))

			# print(points.shape)

			fileStarter = "../inputoutput/matplotfigures/" + metric.__name__ + "_" + kernel.__name__

			meanSolver.plot(colored=True, show=False, name=fileStarter + "_colored.png")
			plt.clf()

			# Histogram
			plt.hist(singlevoltage, bins=100)

			plt.xlabel('Value')
			plt.ylabel('Frequency')
			plt.title('Histogram')

			plt.savefig(fileStarter + '_histogram.png')
			plt.clf()

			# PCA
			pca = PCA(n_components=2)
			points_2d = pca.fit_transform(points)

			# print(points_2d.shape)

			plt.scatter(points_2d[:, 0], points_2d[:, 1], s=10)
			plt.xlabel("PCA Component 1")
			plt.ylabel("PCA Component 2")
			plt.title("PCA Projection of Solver Outputs")

			plt.savefig(fileStarter + "_PCA.png")
			plt.clf()

			# MDS
			mds = MDS(n_components=2, random_state=42)
			transformed_points = mds.fit_transform(points)
			
			plt.figure(figsize=(8, 6))
			plt.scatter(transformed_points[:, 0], transformed_points[:, 1], c='blue', edgecolors='black')
			
			plt.xlabel("MDS Dimension 1")
			plt.ylabel("MDS Dimension 2")
			plt.title("Multidimensional Scaling (MDS) to 2D")
			
			plt.savefig(fileStarter + "_MDS.png")
			plt.clf()