import voltage
import kmeans
import create_data
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Union


dist: List[float] = []

class BestParameterFinder:
	def __init__(self, metric: Optional[Callable[["BestParameterFinder", np.ndarray], float]] = None):
		self.metric = metric or self.expWithStd
		self.p_g: Optional[float] = None
		self.c: Optional[float] = None

	def nInfUniform(self, voltages: np.ndarray) -> float:
		voltages.sort()
		uniform = np.array([x / (len(voltages) - 1) for x in range(len(voltages))])
		return np.linalg.norm(abs(voltages - uniform))

	def nInfExp(self, voltages: np.ndarray, base: float = 10) -> float:
		global dist
		voltages.sort()
		if len(dist) != len(voltages):
			dist = np.array([np.power(base, (x / (len(voltages) - 1)) - 1) for x in range(len(voltages))])
		return np.linalg.norm(abs(voltages - dist))

	def median(self, voltages: np.ndarray, value: float = 0.5) -> float:
		voltages.sort()
		return abs(voltages[int(len(voltages) / 2)] - value)

	def minimum(self, voltages: np.ndarray, value: float = 0.1) -> float:
		voltages.sort()
		return abs(voltages[0] - value)

	def minWithStd(self, voltages: np.ndarray, value: float = 0.1) -> float:
		voltages.sort()
		return abs(voltages[0] - value) / np.std(voltages)

	def expWithStd(self, voltages: np.ndarray, base: float = 10) -> float:
		return self.nInfExp(voltages, base) / np.std(voltages)

	def setResistanceToGround(self, p_g: float) -> None:
		self.p_g = np.log(p_g)

	def setKernelParameter(self, c: float) -> None:
		self.c = np.log(c)

	def calculateFor(
		self,
		landmarks: List,
		data: Union[create_data.Data, kmeans.Partitions],
		c: float,
		p_g: float,
		approx: bool = False,
		approx_epsilon: Optional[float] = None,
		approx_iters: Optional[int] = None
	) -> Union[float, tuple[np.ndarray, voltage.Problem]]:

		if isinstance(data, create_data.Data):
			meanProblem = voltage.Problem(data)
			meanProblem.timeStart()
			meanProblem.setKernel(meanProblem.gaussiankernel)
			meanProblem.setWeights(np.exp(c))

		elif isinstance(data, kmeans.Partitions):
			partitions = data
			meanProblem = voltage.Problem(partitions.centers)
			meanProblem.timeStart()
			meanProblem.setKernel(meanProblem.gaussiankernel)
			meanProblem.setPartitionWeights(partitions, np.exp(c))

		else:
			raise ValueError("Unsupported data type")

		meanProblem.addUniversalGround(np.exp(p_g))
		meanProblem.addLandmarks(landmarks)

		meanProblem.timeEnd()

		if approx:
			voltages = np.array(voltage.Solver(meanProblem).approximate_voltages(approx_epsilon, approx_iters))
		else:
			voltages = np.array(voltage.Solver(meanProblem).compute_voltages())

		meanProblem.timeEnd()

		if self.metric:
			return self.metric(self, voltages)
		else:
			return voltages, meanProblem

	def bestParameterFinder(
		self,
		landmarks: List,
		data: Union[create_data.Data, kmeans.Partitions],
		minBound: float = -25,
		maxBound: float = -1,
		granularity: int = 5,
		epsilon: float = 1,
		approx: Optional[int] = None
	) -> tuple[float, float]:
		"""
		Finds the best parameters (C and P_G) for a solver based on voltage distribution minimization.
		"""
		window_size = (maxBound - minBound) / 2
		bestc = minBound + window_size
		bestg = minBound + window_size
		val = float('inf')

		while window_size > epsilon:
			cs = [bestc + x * window_size / granularity for x in range(-granularity + 1, granularity)]
			gs = [bestg + x * window_size / granularity for x in range(-granularity + 1, granularity)]

			if self.c is not None:
				cs = [self.c]
			if self.p_g is not None:
				gs = [self.p_g]

			for c in cs:
				for g in gs:
					try:
						if approx is None:
							tempval = self.calculateFor(landmarks, data, c, g)
						else:
							tempval = self.calculateFor(landmarks, data, c, g, approx=True, approx_iters=approx)

						if val > tempval:
							bestc, bestg = c, g
							val = tempval
					except ValueError:
						pass

			window_size /= granularity

		return np.exp(bestc), np.exp(bestg)

	def visualizations(self, voltages: List[np.ndarray], fileStarter: str) -> None:
		points = np.array(list(map(list, zip(*voltages))))

		pca = PCA(n_components=2)
		points_2d = pca.fit_transform(points)

		plt.scatter(points_2d[:, 0], points_2d[:, 1], s=10)
		plt.xlabel("PCA Component 1")
		plt.ylabel("PCA Component 2")
		plt.title("PCA Projection of Solver Outputs")
		plt.savefig(fileStarter + "_PCA.png")
		plt.clf()

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