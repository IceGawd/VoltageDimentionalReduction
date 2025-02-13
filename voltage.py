from create_data import *
import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import solve
import threading

class Landmark():
	def __init__(self, index, voltage):
		self.index = index
		self.voltage = voltage

class Solver():
	def __init__(self, data):
		self.data = data
		self.landmarks = []
		n = len(data)
		self.weights = np.zeros([len(data), len(data)])
		self.universalGround = False

	def setWeights(self, kernel, c):
		"""
		for (x, datax) in enumerate(self.data):
			y_iter = enumerate(self.data)

			for y in range(0, x):
				y_iter.__next__()																		# "Burn" x data points in this generator

			for (y, datay) in y_iter:
		"""
		n = len(self.data)

		for x in range(0, n):
			datax = self.data[x]
			for y in range(x, n):
				datay = self.data[y]

				v = kernel(datax, datay, c) / np.pow(n, 2)												# R = n^2/k, W = 1/R = k/n^2

				self.weights[x][y] = v
				self.weights[y][x] = v

		return self.weights

	def setWeights(self, kernel, c, partition):
		n = len(self.data)

		for x in range(0, n):
			datax = self.data[x]
			for y in range(x, n):
				datay = self.data[y]

				v = kernel(datax, datay, c) / (partition.point_counts[x] * partition.point_counts[y])

				self.weights[x][y] = v
				self.weights[y][x] = v

		return self.weights

	def addLandmark(self, landmark):
		self.landmarks.append(landmark)

	def addLandmarks(self, landmarks):
		self.landmarks += landmarks

	def compute_voltages(self):
		n = self.weights.shape[0]
		
		L = csgraph.laplacian(self.weights, normed=False)
		
		constrained_nodes =   [l.index for l in self.landmarks]
		unconstrained_nodes = [i for i in range(n) if i not in constrained_nodes]
		
		b = np.zeros(n)
		for landmark in self.landmarks:
			b[landmark.index] = landmark.voltage
		
		# print(L)
		# print(b)

		L_unconstrained = L[np.ix_(unconstrained_nodes, unconstrained_nodes)]															# Gets diagonal subgraph of L
		# print(L_unconstrained)

		b_unconstrained = b[unconstrained_nodes] - np.matmul(L[np.ix_(unconstrained_nodes, constrained_nodes)], b[constrained_nodes])	# B with the subtraction of the constrained term
		# print(b_unconstrained)
		
		v_unconstrained = solve(L_unconstrained, b_unconstrained)
		
		self.voltages = np.zeros(n)

		for landmark in self.landmarks:
			self.voltages[landmark.index] = landmark.voltage

		self.voltages[unconstrained_nodes] = v_unconstrained
		
		if (self.universalGround):
			self.voltages = self.voltages[:-1]

		return self.voltages

	def addUniversalGround(self):
		self.universalGround = True

		n = self.weights.shape[0]
		newW = np.zeros([n + 1, n + 1])

		newW[0:n,0:n] = self.weights

		p_g = 0.01

		for x in range(0, n):			# W[g, g] = 0
			newW[x][n] = p_g / n
			newW[n][x] = p_g / n

		self.weights = newW
		self.addLandmark(Landmark(n, 0))

		return newW

	def plot(self, color='r', ax=None, show=True, label=""):
		dim = len(self.data[0])

		if (ax == None):
			fig = plt.figure()

			if (dim == 2):
				ax = fig.add_subplot(111, projection='3d')
			else:
				ax = fig.add_subplot(111)
			ax.legend()

		(x_coords, y_coords, z_coords) = pointFormatting(self.data)

		if (dim == 1):
			ax.scatter(x_coords, self.voltages, c=color, marker='o', label=label)
		if (dim == 2):
			ax.scatter(x_coords, y_coords, self.voltages, c=color, marker='o', label=label)

		if (show):
			plt.show()

		return ax

def radialkernel(x, y, r):
	if (distance(x, y) < r):
		return 1
	else:
		return 0

def gaussiankernel(x, y, std):
	return np.exp(np.pow(distance(x, y) / std, 2) / -2)

def bestCFinder(kernel, landmarks, partition, emin=-5, emax=5):
	bestE = emin
	medVol = 0

	for e in range(emin, emax+1):
		# print(e)
		meanSolver = Solver(partition.centers)
		meanSolver.setWeights(kernel=kernel, c=pow(10, e), partition=partition)
		meanSolver.addUniversalGround()
		meanSolver.addLandmarks(landmarks)

		voltages = meanSolver.compute_voltages()
		if (medVol < np.median(voltages)):
			bestE = e
			medVol = np.median(voltages)

	bestV = -10
	medVol = 0
	for v in range(-10, 11):
		# print(v)
		meanSolver = Solver(partition.centers)
		meanSolver.setWeights(kernel=kernel, c=pow(10, bestE) + v * pow(10, bestE - 1), partition=partition)		
		meanSolver.addUniversalGround()
		meanSolver.addLandmarks(landmarks)

		voltages = meanSolver.compute_voltages()
		if (medVol < np.median(voltages)):
			bestV = v
			medVol = np.median(voltages)

	return pow(10, bestE) + bestV * pow(10, bestE - 1)

# Example usage
if __name__ == "__main__":
	data = Data("line.json", stream=True)
	n = len(data)

	ungrounded = Solver(data)

	X0 = []
	X1 = []

	for index, point in enumerate(data):
		if point[0] < 1:
			X0.append(Landmark(index, 0))
		if point[0] > 2:
			X1.append(Landmark(index, 1))

	ungrounded.setWeights(kernel=gaussiankernel, c=0.03)
	ungrounded.addLandmarks(X0)
	ungrounded.addLandmarks(X1)
	voltages = ungrounded.compute_voltages()

	grounded = Solver(data)
	grounded.setWeights(kernel=gaussiankernel, c=0.3)
	grounded.addUniversalGround()
	grounded.addLandmarks(X1)
	grounded_voltage = grounded.compute_voltages()

	ax = ungrounded.plot(color='r', show=False, label="Ungrounded Points")
	ax = grounded.plot(color='b', ax=ax, label="Grounded Points")