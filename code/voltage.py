import create_data
import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import solve
import threading
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Landmark():
	def __init__(self, index, voltage):
		self.index = index
		self.voltage = voltage

def createLandmarkClosestTo(data, point, voltage):
	most_central_index = 0
	mindist = create_data.distance(data[0], point)

	for index in range(1, len(data)):
		dist = create_data.distance(data[index], point)
		if dist < mindist:
			most_central_index = index
			mindist = dist
	
	return Landmark(most_central_index, voltage)


class Solver():
	def __init__(self, data):
		self.data = data
		self.landmarks = []
		n = len(data)
		self.weights = np.zeros([len(data), len(data)])
		self.universalGround = False

	def setWeights(self, kernel, *c):
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

				v = kernel(datax, datay, *c) / np.pow(n, 2)												# R = n^2/k, W = 1/R = k/n^2

				self.weights[x][y] = v
				self.weights[y][x] = v

		return self.weights

	def setPartitionWeights(self, kernel, partition, *c):
		n = len(self.data)
		d = np.pow(len(partition.data), 2)

		for x in range(0, n):
			datax = self.data[x]
			for y in range(x, n):
				datay = self.data[y]

				v = kernel(datax, datay, *c) * (partition.point_counts[x] * partition.point_counts[y]) / d

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

	def addUniversalGround(self, p_g=0.01):
		if (self.universalGround):
			n = self.weights.shape[0] - 1

			for x in range(n):				# W[g, g] = 0
				self.weights[x][n] = p_g / n
				self.weights[n][x] = p_g / n

			return self.weights

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

			return newW

	def localSolver(self, data, partitions, c):
		voltages = [0 for i in range(len(data))]

		for index in range(partitions.k):
			closestIndicies = partitions.getClosestPoints(index)
			closeLandmarksIndicies = []

			for pair in partitions.voronoi.ridge_points:
				if pair[0] == index:
					closeLandmarksIndicies.append(pair[1])
				if pair[1] == index:
					closeLandmarksIndicies.append(pair[0])

			closeLandmarks = []
			for cli in closeLandmarksIndicies:
				closeLandmarks.append(Landmark(cli, self.voltages[cli]))

			localSolver = Solver(data.getSubSet(closestIndicies))
			localSolver.setWeights(gaussiankernel, c)
			localSolver.addLandmarks(closeLandmarks)
			localVoltages = localSolver.compute_voltages()

			for i, v in zip(closestIndicies, localVoltages):
				voltages[i] = v

		return voltages

	def plot(self, color='r', ax=None, show=True, label="", colored=False, name=None):
		dim = len(self.data[0])

		if (ax == None):
			fig = plt.figure()

			if ((dim + (not colored)) == 3):
				ax = fig.add_subplot(111, projection="3d")
			else:
				ax = fig.add_subplot(111)

		if (dim > 3):
			pca = PCA(n_components=2)
			points_2d = pca.fit_transform(self.data)
			x_coords, y_coords, z_coords = points_2d[:, 0], points_2d[:, 1], None

			dim = 2
		else:
			(x_coords, y_coords, z_coords) = create_data.pointFormatting(self.data)


		cmap = None
		c = color
		args = [x_coords, y_coords, z_coords][:dim]
		args.append(self.voltages)
		if colored:
			cmap = 'viridis'
			c = self.voltages
			args = args[:-1]

		# print(c)
		# print(args)
		ax.scatter(*args, c=c, cmap=cmap, marker='o', label=label)

		if (name):
			plt.savefig(name)
		if (show):
			plt.show()

		return ax

def radialkernel(x, y, r):
	if (create_data.distance(x, y) < r):
		return 1
	else:
		return 0

def gaussiankernel(x, y, std):
	return np.exp(np.pow(create_data.distance(x, y) / std, 2) / -2)

# Example usage
if __name__ == "__main__":
	data = create_data.Data("../inputoutput/data/line.json", stream=True)
	n = len(data)

	ungrounded = Solver(data)

	X0 = []
	X1 = []

	for index, point in enumerate(data):
		if point[0] < 1:
			X0.append(Landmark(index, 0))
		if point[0] > 2:
			X1.append(Landmark(index, 1))

	ungrounded.setWeights(gaussiankernel, 0.03)
	ungrounded.addLandmarks(X0)
	ungrounded.addLandmarks(X1)
	voltages = ungrounded.compute_voltages()

	grounded = Solver(data)
	grounded.setWeights(gaussiankernel, 0.3)
	grounded.addUniversalGround()
	grounded.addLandmarks(X1)
	grounded_voltage = grounded.compute_voltages()

	ax = ungrounded.plot(color='r', label="Ungrounded Points")
	ax = grounded.plot(color='b', label="Grounded Points")