import create_data
import kmeans

import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import solve
import threading
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

class Landmark():
	"""Defines at which index datapoint will a voltage be applied to. Indicies could be either partition centers or data points themselves"""
	def __init__(self, index, voltage):
		self.index = index
		self.voltage = voltage

	@staticmethod
	def createLandmarkClosestTo(data, point, voltage, distanceFn=None, ignore=[]):
		if (distanceFn == None):
			distanceFn = kmeans.DistanceBased()

		most_central_index = 0
		mindist = distanceFn.distance(data[0], point)

		for index in range(1, len(data)):
			if (index in ignore):
				continue

			dist = distanceFn.distance(data[index], point)
			if dist < mindist:
				most_central_index = index
				mindist = dist
		
		return Landmark(most_central_index, voltage)

class Problem(kmeans.DistanceBased):
	"""Represents the problem that is trying to be solved"""
	def __init__(self, data):
		self.data = data
		self.landmarks = []
		n = len(data)
		self.weights = np.zeros([len(data), len(data)])
		self.universalGround = False
		super().__init__()

	def timeStart(self):
		self.start = time.time()

	def timeEnd(self, replace=True):
		curTime = time.time()
		diff = curTime - self.start

		if (replace):
			self.start = curTime

		return diff

	def setKernel(self, kernel):
		self.kernel = kernel

	def efficientSquareDistance(self, data):
		data_norm2 = np.sum(data**2, axis=1)

		x_norm2 = data_norm2.reshape(-1, 1)				# shape: (n, 1)
		y_norm2 = data_norm2.reshape(1, -1)				# shape: (1, n)
		return x_norm2 + y_norm2 - 2 * data @ data.T	# shape: (n, n)

	def radialkernel(self, data, r):
		dist2 = self.efficientSquareDistance(data)
		return (dist2 <= r**2).astype(float)
	
	def gaussiankernel(self, data, std):
		dist2 = self.efficientSquareDistance(data)
		return np.exp(-dist2 / (2 * std**2))

	def setWeights(self, *c):
		n = len(self.data)

		data = self.data.getNumpy()

		# print(data.shape)

		self.weights[:n, :n] = self.kernel(data, *c)

		self.normalizeWeights()

		return self.weights

	def normalizeWeights(self):
		self.weights = self.weights / self.weights.sum(axis=1, keepdims=True)

		if np.isnan(self.weights).any():
			raise ValueError("Array contains NaN values!")

	def setPartitionWeights(self, partition, *c):
		n = len(partition.centers)
		centers = np.array(partition.centers)
		counts = np.array(partition.point_counts).reshape(-1, 1)

		K = self.kernel(centers[:, None], centers[None, :], *c)

		W = K * (counts @ counts.T)

		self.weights[:n, :n] = W
		self.normalizeWeights()
		return self.weights

	def addUniversalGround(self, p_g=0.01):
		if (self.universalGround):
			n = self.weights.shape[0] - 1

			for x in range(n):				# W[g, g] = 0
				self.weights[x][n] = p_g / n
				self.weights[n][x] = p_g / n

		else:
			self.universalGround = True

			n = self.weights.shape[0]
			newW = np.zeros([n + 1, n + 1])

			newW[0:n,0:n] = self.weights

			for x in range(0, n):			# W[g, g] = 0
				newW[x][n] = p_g / n
				newW[n][x] = p_g / n

			self.weights = newW
			self.addLandmark(Landmark(n, 0))

		self.normalizeWeights()

		return self.weights

	def addLandmark(self, landmark):
		self.landmarks.append(landmark)

	def addLandmarks(self, landmarks):
		self.landmarks += landmarks

	def addLandmarksInRange(self, minRange, maxRange, voltage):
		adding = []
		for index, point in enumerate(data):
			if np.all(point >= minRange) and np.all(point <= maxRange):
				adding.append(Landmark(index, voltage))

		self.addLandmarks(adding)
		return adding

class Solver(kmeans.DistanceBased):
	"""Solves a given Problem"""
	def __init__(self, problem):
		self.problem = problem
		super().__init__()

	def compute_voltages(self):
		n = self.problem.weights.shape[0]
		
		constrained_nodes =   [l.index for l in self.problem.landmarks]
		unconstrained_nodes = [i for i in range(n) if i not in constrained_nodes]
		
		b = np.zeros(n)
		for landmark in self.problem.landmarks:
			for y in range(0, n):
				b[y] += landmark.voltage * self.problem.weights[y][landmark.index]
		
		A_unconstrained = np.identity(len(unconstrained_nodes)) - self.problem.weights[np.ix_(unconstrained_nodes, unconstrained_nodes)]

		b_unconstrained = b[unconstrained_nodes]

		# print(self.problem.weights)
		# print(A_unconstrained)
		# print(b_unconstrained)
		
		v_unconstrained = solve(A_unconstrained, b_unconstrained)
		
		# print(v_unconstrained)

		self.voltages = np.zeros(n)

		for landmark in self.problem.landmarks:
			self.voltages[landmark.index] = landmark.voltage

		self.voltages[unconstrained_nodes] = v_unconstrained
		
		if (self.problem.universalGround):
			self.voltages = self.voltages[:-1]

		return self.voltages

	def approximate_voltages(self, epsilon=None, max_iters=None):
		n = self.problem.weights.shape[0]

		if (epsilon == None):
			if (max_iters == None):
				epsilon = 1 / n

		constrained_nodes =		[l.index for l in self.problem.landmarks]
		constraints = 			[l.voltage for l in self.problem.landmarks]
		unconstrained_nodes =	[i for i in range(n) if i not in constrained_nodes]

		self.voltages = np.zeros(n)
		voltages = np.zeros(n)

		for landmark in self.problem.landmarks:
			self.voltages[landmark.index] = landmark.voltage

		dist = self.distance(self.voltages, voltages)
		prev_dist = float('inf')

		iterations = 0

		while (((epsilon != None and dist > epsilon * len(self.problem.data)) or (max_iters != None and iterations < max_iters)) and dist < prev_dist):
			voltages = np.matmul(self.problem.weights, self.voltages)
			voltages[constrained_nodes] = constraints
			prev_dist = dist
			dist = self.distance(self.voltages, voltages)

			# print(prev_dist, dist)

			self.voltages = voltages
			iterations += 1

		# print(iterations)

		if (self.problem.universalGround):
			self.voltages = self.voltages[:-1]

		return self.voltages

	def localSolver(self, partitions, c):
		voltages = [0 for i in range(len(self.problem.data))]

		for index in range(partitions.k):
			closestIndicies = partitions.getClosestPoints(index)
			closeproblem.LandmarksIndicies = []

			for pair in partitions.voronoi.ridge_points:
				if pair[0] == index:
					closeproblem.LandmarksIndicies.append(pair[1])
				if pair[1] == index:
					closeproblem.LandmarksIndicies.append(pair[0])

			closeproblem.Landmarks = []
			for cli in closeproblem.LandmarksIndicies:
				closeproblem.Landmarks.append(Landmark(cli, self.voltages[cli]))

			localSolver = Solver(self.problem.data.getSubSet(closestIndicies))
			localSolver.setKernel(self.problem.gaussiankernel)
			localSolver.setWeights(c)
			localSolver.addproblem.Landmarks(closeproblem.Landmarks)
			localVoltages = localSolver.compute_voltages()

			for i, v in zip(closestIndicies, localVoltages):
				voltages[i] = v

		return voltages

# Example usage
if __name__ == "__main__":
	data = create_data.Data("../inputoutput/data/line.json", stream=False)
	n = len(data)

	ungrounded = Problem(data)
	ungrounded.timeStart()
	ungrounded.setKernel(ungrounded.gaussiankernel)
	ungrounded.setWeights(0.3)
	X0 = ungrounded.addLandmarksInRange([0], [1], 0)
	X1 = ungrounded.addLandmarksInRange([2], [3], 1)
	diff1 = ungrounded.timeEnd()

	ungrounded_solver = Solver(ungrounded)
	voltages = ungrounded_solver.compute_voltages()
	diff2 = ungrounded.timeEnd()

	print(diff1)
	print(diff2)

	grounded = Problem(data)
	grounded.setKernel(ungrounded.gaussiankernel)
	grounded.setWeights(0.03)
	grounded.addUniversalGround()
	grounded.addLandmarks(X1)

	grounded_solver = Solver(grounded)
	grounded_voltage = grounded_solver.approximate_voltages(max_iters=100)

	plotter = create_data.Plotter()

	plotter.voltage_plot(ungrounded_solver, color='r', label="Ungrounded Points")
	plotter.voltage_plot(grounded_solver, color='b', label="Grounded Points")