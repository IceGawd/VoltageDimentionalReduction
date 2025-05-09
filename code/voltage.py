import create_data
import kmeans

import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import solve
import threading
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

class Landmark:
    """
    Represents a location in the dataset where a voltage will be applied.

    The `index` can refer either to an individual datapoint or a partition center.
    """

    def __init__(self, index, voltage):
        """
        Initializes a Landmark.

        Args:
            index (int): Index of the datapoint or partition center.
            voltage (float): Voltage to be applied at the specified index.
        """
        self.index = index
        self.voltage = voltage

    @staticmethod
    def createLandmarkClosestTo(data, point, voltage, distanceFn=None, ignore=[]):
        """
        Creates a Landmark at the index of the datapoint in `data` closest to `point`.

        Args:
            data (List[Any]): The dataset to search over.
            point (Any): The reference point to find the closest datapoint to.
            voltage (float): The voltage to assign to the resulting Landmark.
            distanceFn (Optional[object]): A distance function with a `.distance(a, b)` method.
                                           Defaults to `kmeans.DistanceBased()` if None.
            ignore (List[int], optional): List of indices to skip during the search. Defaults to empty list.

        Returns:
            Landmark: A Landmark instance corresponding to the closest datapoint.
        """
        if distanceFn is None:
            distanceFn = kmeans.DistanceBased()

        most_central_index = 0
        mindist = distanceFn.distance(data[0], point)

        for index in range(1, len(data)):
            if index in ignore:
                continue

            dist = distanceFn.distance(data[index], point)
            if dist < mindist:
                most_central_index = index
                mindist = dist

        return Landmark(most_central_index, voltage)

import time
import numpy as np
import kmeans

class Problem(kmeans.DistanceBased):
    """
    Represents the clustering/graph problem to be solved, 
    extending a distance-based kernel with landmarks and weights.
    """

    def __init__(self, data):
        """
        Initializes the Problem instance.

        Args:
            data: An object containing your dataset. Must support len(data) 
                  and data.getNumpy() to return an (n, d) numpy array.
        """
        super().__init__()
        self.data = data
        self.landmarks = []
        n = len(data)
        self.weights = np.zeros([n, n])
        self.universalGround = False

    def timeStart(self):
        """
        Records the current time to measure elapsed intervals.
        """
        self.start = time.time()

    def timeEnd(self, replace=True):
        """
        Computes the elapsed time since the last timeStart().

        Args:
            replace (bool): If True, resets the start time to now.

        Returns:
            float: Seconds elapsed since last start.
        """
        cur_time = time.time()
        diff = cur_time - self.start
        if replace:
            self.start = cur_time
        return diff

    def setKernel(self, kernel):
        """
        Sets the kernel function to use for weight computations.

        Args:
            kernel (callable): A function or callable object with signature
                               kernel(X, Y, *params) → ndarray of shape (|X|, |Y|).
        """
        self.kernel = kernel

    def efficientSquareDistance(self, data):
        """
        Computes the pairwise squared Euclidean distances of the rows in `data`.

        Uses the identity ‖x−y‖² = ‖x‖² + ‖y‖² − 2 x·y for efficiency.

        Args:
            data (ndarray): Array of shape (n, d).

        Returns:
            ndarray: Matrix of shape (n, n) where entry (i, j) is squared distance.
        """
        data_norm2 = np.sum(data**2, axis=1)
        x_norm2 = data_norm2.reshape(-1, 1)
        y_norm2 = data_norm2.reshape(1, -1)
        return x_norm2 + y_norm2 - 2 * data @ data.T

    def radialkernel(self, data, r):
        """
        Builds a binary (0/1) radial kernel: 1 if distance ≤ r, else 0.

        Args:
            data (ndarray): Array of shape (n, d).
            r (float): Radius threshold.

        Returns:
            ndarray: Adjacency-like matrix (n×n) of 0/1 floats.
        """
        dist2 = self.efficientSquareDistance(data)
        return (dist2 <= r**2).astype(float)

    def gaussiankernel(self, data, std):
        """
        Builds a Gaussian (RBF) kernel matrix.

        Args:
            data (ndarray): Array of shape (n, d).
            std (float): Standard deviation parameter for the Gaussian.

        Returns:
            ndarray: Kernel matrix of shape (n, n).
        """
        dist2 = self.efficientSquareDistance(data)
        return np.exp(-dist2 / (2 * std**2))

    def setWeights(self, *c):
        """
        Computes and normalizes the weight matrix on the original data.

        Args:
            *c: Parameters to pass into the currently set kernel function.

        Returns:
            ndarray: The normalized weight matrix (n×n).
        """
        data_np = self.data.getNumpy()
        n = len(self.data)
        self.weights[:n, :n] = self.kernel(data_np, *c)
        self.normalizeWeights()
        return self.weights

    def normalizeWeights(self):
        """
        Normalizes each row of the weight matrix to sum to 1.

        Raises:
            ValueError: If any row sums to zero, resulting in NaNs.
        """
        self.weights = self.weights / self.weights.sum(axis=1, keepdims=True)
        if np.isnan(self.weights).any():
            raise ValueError("Array contains NaN values!")

    def setPartitionWeights(self, partition, *c):
        """
        Computes and normalizes weights based on cluster centers and sizes.

        Args:
            partition: An object with attributes `centers` (list of points)
                       and `point_counts` (counts per center).
            *c: Parameters to pass into the kernel function.

        Returns:
            ndarray: The normalized weight matrix for the partition block.
        """
        centers = np.array(partition.centers)
        counts = np.array(partition.point_counts).reshape(-1, 1)
        K = self.kernel(centers[:, None], centers[None, :], *c)
        W = K * (counts @ counts.T)
        n = len(centers)
        self.weights[:n, :n] = W
        self.normalizeWeights()
        return self.weights

    def addUniversalGround(self, p_g=0.01):
        """
        Adds (or updates) a 'universal ground' node connected uniformly to all others.

        Args:
            p_g (float): Total ground connection probability to distribute.

        Returns:
            ndarray: The updated normalized weight matrix including the ground node.
        """
        if self.universalGround:
            n = self.weights.shape[0] - 1
            for x in range(n):
                self.weights[x, n] = p_g / n
                self.weights[n, x] = p_g / n
        else:
            self.universalGround = True
            n = self.weights.shape[0]
            newW = np.zeros([n + 1, n + 1])
            newW[:n, :n] = self.weights
            for x in range(n):
                newW[x, n] = p_g / n
                newW[n, x] = p_g / n
            self.weights = newW
            self.addLandmark(Landmark(n, 0))
        self.normalizeWeights()
        return self.weights

    def addLandmark(self, landmark):
        """
        Adds a single Landmark to the problem.

        Args:
            landmark (Landmark): The landmark instance to append.
        """
        self.landmarks.append(landmark)

    def addLandmarks(self, landmarks):
        """
        Adds multiple Landmark instances to the problem.

        Args:
            landmarks (List[Landmark]): List of landmarks to append.
        """
        self.landmarks += landmarks

    def addLandmarksInRange(self, minRange, maxRange, voltage):
        """
        Adds landmarks for all data points within a given coordinate range.

        Args:
            minRange (array-like): Minimum bounds per dimension.
            maxRange (array-like): Maximum bounds per dimension.
            voltage (float): Voltage to apply at each new landmark.

        Returns:
            List[Landmark]: The list of newly added landmarks.
        """
        adding = []
        data_np = self.data.getNumpy()
        for idx, point in enumerate(data_np):
            if np.all(point >= minRange) and np.all(point <= maxRange):
                adding.append(Landmark(idx, voltage))
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