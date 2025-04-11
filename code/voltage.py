import create_data
import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import solve
import threading
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

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
		n = len(self.data)

		data = self.data.getNumpy()

		self.weights[:n, :n] = kernel(data[None, :, :], data[:, None, :], *c)

		self.normalizeWeights()

		return self.weights

	def normalizeWeights(self):
		self.weights = self.weights / self.weights.sum(axis=1, keepdims=True)

		if np.isnan(self.weights).any():
			raise ValueError("Array contains NaN values!")

	def setPartitionWeights(self, kernel, partition, *c):
		n = len(partition.centers)
		centers = np.array(partition.centers)
		counts = np.array(partition.point_counts).reshape(-1, 1)

		K = kernel(centers[:, None], centers[None, :], *c)

		W = K * (counts @ counts.T)

		self.weights[:n, :n] = W
		self.normalizeWeights()
		return self.weights

	def addLandmark(self, landmark):
		self.landmarks.append(landmark)

	def addLandmarks(self, landmarks):
		self.landmarks += landmarks

	def compute_voltages(self):
		n = self.weights.shape[0]
		
		constrained_nodes =   [l.index for l in self.landmarks]
		unconstrained_nodes = [i for i in range(n) if i not in constrained_nodes]
		
		b = np.zeros(n)
		for landmark in self.landmarks:
			for y in range(0, n):
				b[y] += landmark.voltage * self.weights[y][landmark.index]
		
		A_unconstrained = np.identity(len(unconstrained_nodes)) - self.weights[np.ix_(unconstrained_nodes, unconstrained_nodes)]

		b_unconstrained = b[unconstrained_nodes]

		# print(self.weights)
		# print(A_unconstrained)
		# print(b_unconstrained)
		
		v_unconstrained = solve(A_unconstrained, b_unconstrained)
		
		# print(v_unconstrained)

		self.voltages = np.zeros(n)

		for landmark in self.landmarks:
			self.voltages[landmark.index] = landmark.voltage

		self.voltages[unconstrained_nodes] = v_unconstrained
		
		if (self.universalGround):
			self.voltages = self.voltages[:-1]

		return self.voltages

	def approximate_voltages(self, epsilon=None, max_iters=None):
		n = self.weights.shape[0]

		if (epsilon == None):
			if (max_iters == None):
				epsilon = 1 / n

		constrained_nodes =		[l.index for l in self.landmarks]
		constraints = 			[l.voltage for l in self.landmarks]
		unconstrained_nodes =	[i for i in range(n) if i not in constrained_nodes]

		self.voltages = np.zeros(n)
		voltages = np.zeros(n)

		for landmark in self.landmarks:
			self.voltages[landmark.index] = landmark.voltage

		dist = create_data.distance(self.voltages, voltages)
		prev_dist = float('inf')

		iterations = 0

		while (((epsilon != None and dist > epsilon * len(self.data)) or (max_iters != None and iterations < max_iters)) and dist < prev_dist):
			voltages = np.matmul(self.weights, self.voltages)
			voltages[constrained_nodes] = constraints
			prev_dist = dist
			dist = create_data.distance(self.voltages, voltages)

			# print(prev_dist, dist)

			self.voltages = voltages
			iterations += 1

		# print(iterations)

		if (self.universalGround):
			self.voltages = self.voltages[:-1]

		return self.voltages

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
	return (np.linalg.norm(diffs, axis=-1) <= r).astype(float)

def gaussiankernel(x, y, std):
	return np.exp(-(np.linalg.norm(x - y, axis=-1) ** 2) / (2 * std ** 2))

# Example usage
if __name__ == "__main__":
	data = create_data.Data("../inputoutput/data/line.json", stream=False)
	n = len(data)

	ungrounded = Solver(data)

	X0 = []
	X1 = []

	for index, point in enumerate(data):
		if point[0] < 1:
			X0.append(Landmark(index, 0))
		if point[0] > 2:
			X1.append(Landmark(index, 1))

	# """
	start = time.time()

	ungrounded.setWeights(gaussiankernel, 0.3)
	ungrounded.addLandmarks(X0)
	ungrounded.addLandmarks(X1)
	# voltages = ungrounded.compute_voltages()

	end = time.time()

	voltages = ungrounded.approximate_voltages(max_iters=100)
	# """

	end2 = time.time()

	print(end - start)
	print(end2 - end)

	grounded = Solver(data)
	grounded.setWeights(gaussiankernel, 0.03)
	grounded.addUniversalGround()
	grounded.addLandmarks(X1)
	# grounded_voltage = grounded.compute_voltages()
	grounded_voltage = grounded.approximate_voltages(max_iters=100)

	ax = ungrounded.plot(color='r', label="Ungrounded Points")
	ax = grounded.plot(color='b', label="Grounded Points")

	# ax = ungrounded.plot(color='r', label="Ungrounded Points", name="../inputoutput/matplotfigures/approxUngrounded.png")
	# ax = grounded.plot(color='b', label="Grounded Points", name="../inputoutput/matplotfigures/approxGrounded.png")