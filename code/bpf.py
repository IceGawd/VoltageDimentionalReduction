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
        """
        Initializes the BestParameterFinder.

        Args:
            metric (Optional[Callable[[BestParameterFinder, np.ndarray], float]]):
                A custom metric function. Defaults to `expWithStd`.
        """
        self.metric = metric or self.expWithStd
        self.p_g: Optional[float] = None
        self.c: Optional[float] = None

    def nInfUniform(self, voltages: np.ndarray) -> float:
        """
        Computes the infinity-norm distance between voltages and a uniform distribution.

        Args:
            voltages (np.ndarray): Array of voltage values.

        Returns:
            float: Infinity-norm distance.
        """
        voltages.sort()
        uniform = np.array([x / (len(voltages) - 1) for x in range(len(voltages))])
        return np.linalg.norm(abs(voltages - uniform))

    def nInfExp(self, voltages: np.ndarray, base: float = 10) -> float:
        """
        Computes the infinity-norm distance between voltages and an exponential distribution.

        Args:
            voltages (np.ndarray): Array of voltage values.
            base (float): Base of the exponential function. Defaults to 10.

        Returns:
            float: Infinity-norm distance.
        """
        global dist
        voltages.sort()
        if len(dist) != len(voltages):
            dist = np.array([np.power(base, (x / (len(voltages) - 1)) - 1) for x in range(len(voltages))])
        return np.linalg.norm(abs(voltages - dist))

    def median(self, voltages: np.ndarray, value: float = 0.5) -> float:
        """
        Computes the absolute difference between the median voltage and a given value.

        Args:
            voltages (np.ndarray): Array of voltage values.
            value (float): Value to compare the median to. Defaults to 0.5.

        Returns:
            float: Absolute difference from the median.
        """
        voltages.sort()
        return abs(voltages[int(len(voltages) / 2)] - value)

    def minimum(self, voltages: np.ndarray, value: float = 0.1) -> float:
        """
        Computes the absolute difference between the minimum voltage and a given value.

        Args:
            voltages (np.ndarray): Array of voltage values.
            value (float): Value to compare the minimum to. Defaults to 0.1.

        Returns:
            float: Absolute difference from the minimum.
        """
        voltages.sort()
        return abs(voltages[0] - value)

    def minWithStd(self, voltages: np.ndarray, value: float = 0.1) -> float:
        """
        Computes the normalized difference between the minimum voltage and a given value.

        Args:
            voltages (np.ndarray): Array of voltage values.
            value (float): Value to compare the minimum to. Defaults to 0.1.

        Returns:
            float: Normalized absolute difference using standard deviation.
        """
        voltages.sort()
        return abs(voltages[0] - value) / np.std(voltages)

    def expWithStd(self, voltages: np.ndarray, base: float = 10) -> float:
        """
        Computes the normalized exponential distance.

        Args:
            voltages (np.ndarray): Array of voltage values.
            base (float): Base of the exponential. Defaults to 10.

        Returns:
            float: Normalized exponential distance.
        """
        return self.nInfExp(voltages, base) / np.std(voltages)

    def setResistanceToGround(self, p_g: float) -> None:
        """
        Sets the resistance to ground parameter.

        Args:
            p_g (float): Resistance to ground value (logarithmic scale will be used).
        """
        self.p_g = np.log(p_g)

    def setKernelParameter(self, c: float) -> None:
        """
        Sets the kernel parameter.

        Args:
            c (float): Kernel parameter (logarithmic scale will be used).
        """
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
        """
        Calculates voltages and applies the metric.

        Args:
            landmarks (List): Landmarks to add to the problem.
            data (Union[create_data.Data, kmeans.Partitions]): Input data.
            c (float): Kernel parameter (log space).
            p_g (float): Resistance to ground (log space).
            approx (bool): Whether to use approximation. Defaults to False.
            approx_epsilon (Optional[float]): Epsilon value for approximation.
            approx_iters (Optional[int]): Number of approximation iterations.

        Returns:
            Union[float, tuple[np.ndarray, voltage.Problem]]: Metric value or voltages and problem.
        """

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
        Finds optimal (C, P_G) parameters minimizing the metric.

        Args:
            landmarks (List): Landmarks to use in solving.
            data (Union[create_data.Data, kmeans.Partitions]): Input dataset.
            minBound (float): Minimum log-bound for search. Defaults to -25.
            maxBound (float): Maximum log-bound for search. Defaults to -1.
            granularity (int): Granularity of grid search. Defaults to 5.
            epsilon (float): Precision threshold. Defaults to 1.
            approx (Optional[int]): Approximation iteration count. Defaults to None.

        Returns:
            tuple[float, float]: Best (C, P_G) parameters (in real scale).
        """

    def visualizations(self, voltages: List[np.ndarray], fileStarter: str) -> None:
        """
        Generates and saves PCA and MDS visualizations of the voltage data.

        Args:
            voltages (List[np.ndarray]): List of voltage arrays.
            fileStarter (str): File name prefix for saving plots.

        Returns:
            None
        """


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