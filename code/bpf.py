import voltage
import kmeans
import create_data
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

dist = []

class BestParameterFinder():
	def nInfUniform(self, voltages):
		voltages.sort()
		uniform = np.array([x / (len(voltages) - 1) for x in range(len(voltages))])

		return np.linalg.norm(abs(voltages - uniform))

	def nInfExp(self, voltages, base=10):
		global dist

		voltages.sort()

		if (len(dist) != len(voltages)):
			dist = np.array([np.pow(base, (x / (len(voltages) - 1)) - 1) for x in range(len(voltages))])

		return np.linalg.norm(abs(voltages - dist))

	def median(self, voltages, value=0.5):
		voltages.sort()
		return abs(voltages[int(len(voltages) / 2)] - value)

	def minimum(self, voltages, value=0.1):
		voltages.sort()
		return abs(voltages[0] - value)

	def minWithStd(self, voltages, value=0.1):
		voltages.sort()
		return abs(voltages[0] - value) / np.std(voltages)

	def __init__(self, metric=nInfUniform):
		self.metric = metric

	def calculateFor(self, landmarks, data, c, p_g, approx=False, approx_epsilon=None, approx_iters=None):
		# print(type(data))

		if (isinstance(data, create_data.Data)):
			meanProblem = voltage.Problem(data)
			meanProblem.timeStart()
			meanProblem.setKernel(meanProblem.gaussiankernel)
			# print("before")
			meanProblem.setWeights(np.exp(c))
			# print("after")
			# print(meanProblem)

		if (isinstance(data, kmeans.Partitions)):
			partitions = data

			meanProblem = voltage.Problem(partition.centers)
			meanProblem.timeStart()
			meanProblem.setKernel(meanProblem.gaussiankernel)
			meanProblem.setPartitionWeights(partition, np.exp(c))
			# print(meanProblem)

		# print(meanProblem)

		meanProblem.addUniversalGround(np.exp(p_g))
		meanProblem.addLandmarks(landmarks)

		diff1 = meanProblem.timeEnd()
		# print(diff1)

		if (approx):
			voltages = np.array(voltage.Solver(meanProblem).approximate_voltages(approx_epsilon, approx_iters))
		else:
			voltages = np.array(voltage.Solver(meanProblem).compute_voltages())

		diff2 = meanProblem.timeEnd()
		# print(diff2)

		if (self.metric):
			return self.metric(self, voltages)
		else:
			return voltages, meanProblem

	def bestParameterFinder(self, landmarks, data, minBound=-25, maxBound=-1, granularity=5, epsilon=1, approx=None):
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
		data : object
			Either a data object or a partition object containing centers used in the solver.
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
					# print(c, g)
					try:
						if (approx == None):
							tempval = self.calculateFor(landmarks, data, c, g)
						else:
							tempval = self.calculateFor(landmarks, data, c, g, approx=True, approx_iters=approx)							

						# print(tempval)
						if (val > tempval):
							# print(c)
							# print(g)
							# print(tempval)

							bestc = c
							bestg = g
							val = tempval
					except ValueError as e:
						pass
						# print("Invalid")


			window_size /= granularity

		return np.exp(bestc), np.exp(bestg)

	def visualizations(self, voltages, fileStarter):
		points = np.array(list(map(list, zip(*voltages))))

		# print(points.shape)

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

if __name__ == "__main__":
	print("Loading data...")
	data = create_data.Data("../inputoutput/data/spiral.json", stream=True)

	print("Doing partitioning...")
	k = 100

	partitions = kmeans.Partitions(data)
	partitions.k_means(k, seed=time.time())

	landmarks = [voltage.createLandmarkClosestTo(partitions.centers, [0, 0], 1)]
	allLandmarks = [voltage.createLandmarkClosestTo(partitions.centers, [1, 1], 1), voltage.createLandmarkClosestTo(partitions.centers, [2, 0], 1)]

	for metric in [nInfUniform, nInfExp, median, minimum, minWithStd]:
		for kernel in [voltage.gaussiankernel]:
			c, p_g = bestParameterFinder(kernel, landmarks, partitions, metric=metric)
			print("Metric: " + metric.__name__ + " Kernel: " + kernel.__name__ + " c: " + str(np.exp(c)) + " p_g: " + str(np.exp(p_g)))

			fileStarter = "../inputoutput/matplotfigures/" + metric.__name__ + "_" + kernel.__name__

			singlevoltage, meanProblem = calculateFor(kernel, landmarks, partitions, None, c, p_g)
			# print(singlevoltage)

			voltages = [singlevoltage]
			for landmark in allLandmarks:
				voltages.append(calculateFor(kernel, [landmark], partitions, None, c, p_g)[0])

			meanProblem.plot(colored=True, show=False, name=fileStarter + "_colored.png")
			plt.clf()

			# Histogram
			plt.hist(singlevoltage, bins=100)

			plt.xlabel('Value')
			plt.ylabel('Frequency')
			plt.title('Histogram')

			plt.savefig(fileStarter + '_histogram.png')
			plt.clf()

			visualizations(voltages, fileStarter)