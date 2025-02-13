from create_data import *
import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import solve
import threading

class Landmark():
	def __init__(self, index, voltage):
		self.index = index
		self.voltage = voltage

class LVM():
	def __init__(self, data):
		self.data = data
		self.landmarks = []

	def setWeights(self, kernel, c):
		n = len(self.data)

		self.weights = np.zeros([n, n])
		"""
		for (x, datax) in enumerate(self.data):
			y_iter = enumerate(self.data)

			for y in range(0, x):
				y_iter.__next__()							# "Burn" x data points in this generator

			for (y, datay) in y_iter:
		"""
		for x in range(0, n):
			datax = self.data[x]
			for y in range(x, n):
				datay = self.data[y]

				v = kernel(datax, datay, c) / np.pow(n, 2)	# R = n^2/k, W = 1/R = k/n^2

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
		
		return self.voltages

	def addGround(self):
		n = self.weights.shape[0]
		newW = np.zeros([n + 1, n + 1])

		newW[0:n,0:n] = self.weights

		p_g = 0.01

		for x in range(0, n):			# W[g, g] = 0
			newW[x][n] = p_g / n
			newW[n][x] = p_g / n

		return newW

def radialkernel(x, y, r):
	if (distance(x, y) < r):
		return 1
	else:
		return 0

def gaussiankernel(x, y, std):
	return np.exp(np.pow(distance(x, y) / std, 2) / -2)

# Example usage
if __name__ == "__main__":
	data = Data("line.json")
	n = len(data)

	ungrounded = LVM(data)

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

	grounded = LVM(data)
	grounded.setWeights(kernel=gaussiankernel, c=0.3)
	grounded.weights = grounded.addGround()
	grounded.addLandmarks(X1)
	grounded_voltage = grounded.compute_voltages()
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter([x[0] for x in data], voltages, c='r', marker='o', label='Voltage Points')
	ax.scatter([x[0] for x in data], grounded_voltage[:-1], c='b', marker='.', label='Grounded Points')
	ax.legend()
	plt.show()