import numpy as np
import json
import ijson
import pickle
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
import time
import types

def select_random(array: list) -> any:
	"""
	Selects a random element from an array.

	Args:
		array (list): The array to select from.

	Returns:
		Any: A random element from the array.
	"""
	return array[int(len(array) * random.random())]


def dimentional_variation(dimentions: int) -> np.ndarray:
	"""
	Returns a NumPy array of random values from a standard normal distribution.

	Args:
		dimentions (int): Number of dimensions/values to return.

	Returns:
		np.ndarray: Array of random values sampled from the standard normal distribution.
	"""
	z_vals = []
	for d in range(dimentions):
		z_vals.append(stats.norm.ppf(random.random()))

	return np.array(z_vals)


def varied_point(mean: np.ndarray, std: float) -> np.ndarray:
	"""
	Returns a point that is randomly offset from the mean based on standard deviation.

	Args:
		mean (np.ndarray): The mean location of the point.
		std (float): Standard deviation to apply.

	Returns:
		np.ndarray: A randomly varied point.
	"""
	return mean + std * dimentional_variation(len(mean))

class Plotter:
	"""
	Graphs the data into different formats.
	"""

	def pointFormatting(self, points: list[np.ndarray]) -> tuple[list[float], list[float], Optional[list[float]]]:
		"""
		Formats points into separate coordinate lists for plotting.

		Args:
			points (list[np.ndarray]): A list of points as NumPy arrays.

		Returns:
			tuple: x, y, and optionally z coordinate lists.
		"""
		size = len(points[0])
		x_coords = [point[0] for point in points]
		z_coords = None
		if size > 1:
			y_coords = [point[1] for point in points]
			if size > 2:
				z_coords = [point[2] for point in points]
		else:
			y_coords = [0 for point in points]
		return (x_coords, y_coords, z_coords)

	def plotPoints(self, points: list[np.ndarray], name: Optional[str] = None) -> None:
		"""
		Plots a single set of points in 2D or 3D.

		Args:
			points (list[np.ndarray]): A list of points to plot.
			name (Optional[str]): Optional filename to save the plot.
		"""
		self.plotPointSets([points], name)

	def plotPointSets(self, sets: list[list[np.ndarray]], name: Optional[str] = None) -> None:
		"""
		Plots multiple sets of points in different colors.

		Args:
			sets (list[list[np.ndarray]]): A list of point sets.
			name (Optional[str]): Optional filename to save the plot.
		"""
		markers = ['o', 'v', '*']
		color = ['r', 'g', 'b']
		size = len(sets[0][0])
		fig = plt.figure()
		if size == 3:
			ax = fig.add_subplot(111, projection='3d')
		else:
			ax = fig.add_subplot(111)
		for i, points in enumerate(sets):
			(x_coords, y_coords, z_coords) = self.pointFormatting(points)
			if size == 3:
				ax.scatter(x_coords, y_coords, z_coords, c=color[i], marker=markers[i], label='Points')
			else:
				ax.scatter(x_coords, y_coords, c=color[i], marker=markers[i], label='Points')
		ax.legend()
		if name:
			plt.savefig(name)
		plt.show()

	def voltage_plot(
		self,
		solver,
		color: str = 'r',
		ax = None,
		show: bool = True,
		label: str = "",
		colored: bool = False,
		name: Optional[str] = None
	):
		"""
		Plots voltage data overlaid on input data using optional PCA projection.

		Args:
			solver: A voltage solver instance with `.problem.data` and `.voltages`.
			color (str): Color for the points if `colored` is False.
			ax: Matplotlib axis to plot on (if provided).
			show (bool): Whether to show the plot.
			label (str): Label for the legend.
			colored (bool): Whether to color the points by voltage values.
			name (Optional[str]): Optional filename to save the plot.

		Returns:
			The axis with the plotted data.
		"""
		dim = len(solver.problem.data[0])

		if ax is None:
			fig = plt.figure()
			if (dim + (not colored)) == 3:
				ax = fig.add_subplot(111, projection="3d")
			else:
				ax = fig.add_subplot(111)

		if dim > 3:
			pca = PCA(n_components=2)
			points_2d = pca.fit_transform(solver.problem.data)
			x_coords, y_coords, z_coords = points_2d[:, 0], points_2d[:, 1], None
			dim = 2
		else:
			x_coords, y_coords, z_coords = self.pointFormatting(solver.problem.data)

		cmap = None
		c = color
		args = [x_coords, y_coords, z_coords][:dim]
		args.append(solver.voltages)

		if colored:
			cmap = 'viridis'
			c = solver.voltages
			args = args[:-1]

		ax.scatter(*args, c=c, cmap=cmap, marker='o', label=label)

		if name:
			plt.savefig(name)
		if show:
			plt.show()

		return ax

class Data():
	"""Class for handling and processing data sets."""
	def __init__(self, arg=None, stream=False):
		self.stream = stream

		if isinstance(arg, list):
			self.data = np.array(arg)
			self.length = len(self.data)
		elif isinstance(arg, str):
			if (stream):
				self.data = self.stream_data_json(arg)
				self.length = next(self.data)
				self.i = 0
			else:
				self.load_data(arg)
				self.length = len(self.data)

			self.input_file = arg
		else:
			self.data = arg
			self.length = len(self.data)

	# Len can run on Data
	def __len__(self):
		return self.length

	# Data can be get indexed
	def __getitem__(self, index):
		if (self.stream):
			if (index < self.i):
				self.data = self.stream_data_json(self.input_file)
				next(self.data)
				self.i = 0

			while (self.i <= index):
				value = next(self.data)
				self.i += 1

			return value
		else:
			return self.data[index]

	# """
	def __setitem__(self, index, value):
		self.data[index] = value
	# """

	# Make Data able to be for looped
	def __iter__(self):
		if (hasattr(self, 'input_file')):
			self.streaming_data = self.stream_data_json(self.input_file)
			next(self.streaming_data)
		else:
			self.streaming_data = 0

		return self

	def __next__(self):
		try:
			if (hasattr(self, 'input_file')):
				return np.array(next(self.streaming_data))
			else:
				if (self.streaming_data == self.length):
					raise
				else:
					return np.array(self.data[self.streaming_data])

				self.streaming_data += 1
		except StopIteration:
			raise

	def getSubSet(self, indexList):
		"""Returns a subset of the data given a list of indices."""
		subset = []
		for index in indexList:
			subset.append(self.data[index])
		return Data(subset)

	def save_data_json(self, output_file):
		fg = FileGenerator()
		fg.setGenerator(fg.linear_generator)
		fg.stream_save(output_file, self.data)

	def save_data_pickle(self, output_file):
		with open(output_file, 'wb') as f: 
			pickle.dump(self.data, f) 

	def load_data_json(self, input_file):
		with open(input_file, 'r') as f:
			self.input_file = input_file

			data = json.load(f)
			self.data = data["data"]
			self.length = data["length"]
			for i, point in enumerate(self.data):
				self.data[i] = np.array(point)

			return self.data

	def load_data_pickle(self, input_file):
		with open(input_file, 'r') as f:
			self.input_file = input_file
			self.data = pickle.load(f)

			return self.data

	def stream_data_json(self, input_file):
		"""Stream the dataset if its saved in a json file"""
		with open(input_file, 'rb') as f:
			f.seek(0, 2)
			position = f.tell()

			value = ""
			read = False
			while position > 0:
				position -= 1
				f.seek(position)
				byte = f.read(1)

				if byte == b' ':
					# print(value)
					yield int(value)
					break

				if (read):
					value = byte.decode() + value

				if byte == b'}':
					read = True

		with open(input_file, 'r') as f:
			f.readline()

			for line in f:
				if ("length" in line):
					break

				data = json.loads(line.strip().split(']')[0] + ']')
				yield np.array(data)

	file_function_pairs = [["json", save_data_json, load_data_json], ["pkl", save_data_pickle, load_data_pickle]]

	def data_function(self, file, save_or_load):
		"""Used for saving and loading the dataset"""
		if (file == None):
			return

		for ffp in self.file_function_pairs:
			if file[-len(ffp[0]):] == ffp[0]:
				if save_or_load == 1:
					ffp[save_or_load](self.data, file)
				else:
					return ffp[save_or_load](self, file)

	def save_data(self, output_file):
		self.data_function(output_file, 1)
		return self

	def load_data(self, input_file):
		self.data_function(input_file, 2)
		return self

	def get_random_point(self):
		"""Returns a random point from the dataset."""
		return select_random(self.data)

	def plot(self, name=None):
		"""Plots the dataset."""
		Plotter().plotPoints(self.data, name)

	def getNumpy(self):
		if isinstance(self.data, np.ndarray):
			# print(self.data.shape)
			return self.data
		else:
			temp = []
			for x in self.data:
				temp.append(np.array(x))

			# print(np.array(temp).shape)
			return np.array(temp)

class FileGenerator:
    """
    Generates files for saved data.

    This class is designed to assist in saving generated datasets in a streaming
    fashion. It provides several built-in generators to create synthetic datasets
    for use with `Data` and `DataCreator` classes.
    """

    def __init__(self):
        """Initializes the FileGenerator."""
        pass

    def setGenerator(self, fn):
        """
        Sets the generator function to be used when saving data.

        Args:
            fn (Callable): A generator function that yields data points.
        """
        self.data_generator = fn

    def stream_save(self, output_file: str, *args):
        """
        Saves data to a JSON file in a streaming manner.

        Args:
            output_file (str): Path to the file where data will be saved.
            *args: Arguments to pass to the generator function.

        Returns:
            None
        """
        with open(output_file, "w") as f:
            f.write("{\"data\": [\n")
            first = True
            length = 0
            for array in self.data_generator(*args):
                if not first:
                    f.write(", \n")
                json.dump(list(array), f)
                length += 1
                first = False
            f.write("], \n\"length\": " + str(length) + "}")

    def linear_generator(self, data: np.ndarray):
        """
        Yields data points one by one from a NumPy array.

        Args:
            data (np.ndarray): Input data.

        Yields:
            np.ndarray: Single data points from the array.
        """
        for d in data.tolist():
            yield d

    def line_generator(self, start: float, end: float, points: int):
        """
        Generates points along a line in 1D space.

        Args:
            start (float): Starting point of the line.
            end (float): Ending point of the line.
            points (int): Number of points to generate.

        Yields:
            np.ndarray: Single-point arrays sampled along the line.
        """
        for _ in range(points):
            yield np.array([random.random() * (end - start) + start])

    def eigth_sphere_generator(self, radius: float, x_pos: int, y_pos: int, z_pos: int, points: int):
        """
        Generates points on an eighth of a sphere surface.

        Args:
            radius (float): Radius of the sphere.
            x_pos (int): Hemisphere direction for X (0 or 1).
            y_pos (int): Hemisphere direction for Y (0 or 1).
            z_pos (int): Hemisphere direction for Z (0 or 1).
            points (int): Number of points to generate.

        Yields:
            np.ndarray: Points on the eighth sphere surface.
        """
        for _ in range(points):
            z = random.random()
            angleXY = np.pi * random.random() / 2
            yield np.array([
                radius * np.sqrt(1 - z**2) * np.cos(angleXY) * (2 * x_pos - 1),
                radius * np.sqrt(1 - z**2) * np.sin(angleXY) * (2 * y_pos - 1),
                radius * z * (2 * z_pos - 1)
            ])

    def triangle_generator(self, edges: list, points: int):
        """
        Generates points uniformly within a triangle defined by three vertices.

        Args:
            edges (list): A list of three points (each a list or np.ndarray) defining the triangle.
            points (int): Number of points to generate.

        Yields:
            np.ndarray: Points uniformly sampled inside the triangle.
        """
        base = np.array(edges[0])
        edgeDiff1 = np.array(edges[1]) - base
        edgeDiff2 = np.array(edges[2]) - base
        for _ in range(points):
            d1 = random.random()
            d2 = random.random()
            if d1 + d2 > 1:
                d1 = 1 - d1
                d2 = 1 - d2
            yield base + d1 * edgeDiff1 + d2 * edgeDiff2

    def strong_cluster_generator(self, internal_std: float, cluster_centers: list, points: int):
        """
        Generates clustered points around multiple centers with specified standard deviation.

        Args:
            internal_std (float): Standard deviation within each cluster.
            cluster_centers (list): A list of cluster center points.
            points (int): Number of points to generate.

        Yields:
            np.ndarray: Points sampled from the clusters.
        """
        c = -1
        for p in range(points):
            if (p / points >= c / 100):
                c += 1
            yield varied_point(select_random(cluster_centers), internal_std)

    def spiral_generator(self, radius: float, center: list, rotations: int, height: float, points: int):
        """
        Generates points forming a 3D spiral (helix).

        Args:
            radius (float): Radius of the spiral.
            center (list): Center offset of the spiral (not used directly in current implementation).
            rotations (int): Number of full 360° turns.
            height (float): Total height of the spiral.
            points (int): Number of points to generate.

        Yields:
            np.ndarray: Points along the spiral.
        """
        line = 2 * np.pi * rotations
        heightPerRadian = height / line
        for _ in range(points):
            d = random.random() * line
            yield np.array([
                radius * np.cos(d),
                radius * np.sin(d),
                heightPerRadian * d
            ])

class DataCreator:
    """
    A utility class to create various synthetic datasets for testing and analysis.
    Interfaces with FileGenerator to optionally stream data to file.

    Attributes:
        fg (FileGenerator): An instance of FileGenerator used for generating data points.
    """

    def __init__(self):
        self.fg = FileGenerator()

    def stream_dataset_creator(self, output_file: str, function: callable, seed: int, stream: bool, *args) -> 'Data':
        """
        Creates a dataset using the specified generator function, supporting streamed or non-streamed output.

        Args:
            output_file (str): File path to save the dataset.
            function (callable): Generator function to create data points.
            seed (int): Random seed for reproducibility.
            stream (bool): If True, streams data directly to the file.
            *args: Additional arguments passed to the generator function.

        Returns:
            Data: The created dataset, either streamed or in-memory.
        """
        random.seed(seed)

        if stream:
            self.fg.setGenerator(function)
            self.fg.stream_save(output_file, *args)
            data = Data(output_file, stream=True)
        else:
            data = [point for point in function(*args)]
            data = Data(data)
            data.save_data(output_file)

        return data

    def create_dataset_line(self, output_file: str = None, start: float = 0, end: float = 1, points: int = 1000, seed: int = 42, stream: bool = False) -> 'Data':
        """
        Creates a 1D line dataset.

        Args:
            output_file (str): File path to save the dataset.
            start (float): Starting point of the line.
            end (float): Ending point of the line.
            points (int): Number of data points.
            seed (int): Random seed.
            stream (bool): Whether to stream to file.

        Returns:
            Data: The generated dataset.
        """
        return self.stream_dataset_creator(output_file, self.fg.line_generator, seed, stream, start, end, points)

    def create_dataset_square_edge(self, output_file: str = None, p1: tuple = (0, 0), p2: tuple = (1, 1), points: int = 1000, seed: int = 42) -> 'Data':
        """
        Creates a dataset of points along the edges of a square.

        Args:
            output_file (str): File path to save the dataset.
            p1 (tuple): Bottom-left corner.
            p2 (tuple): Top-right corner.
            points (int): Number of data points.
            seed (int): Random seed.

        Returns:
            Data: The generated dataset.
        """
        data = []
        random.seed(seed)

        x_diff = p2[0] - p1[0]
        y_diff = p2[1] - p1[1]

        for _ in range(points):
            r = random.random() * 4
            side = int(r)
            var = r - side

            x_side = side % 2
            y_side = side >> 1

            x_rev = 1 - x_side
            y_rev = 1 - y_side

            variation = np.array([var * x_side * x_diff, var * x_rev * y_diff])
            offset = np.array([x_rev * y_side * x_diff, x_side * y_rev * y_diff])
            shift = np.array(p1)

            data.append(variation + offset + shift)

        data = Data(data)
        data.save_data(output_file)
        return data

    def create_dataset_square_fill(self, output_file: str = None, p1: tuple = (0, 0), p2: tuple = (1, 1), points: int = 1000, seed: int = 42) -> 'Data':
        """
        Creates a dataset of points filling a square area.

        Args:
            output_file (str): File path to save the dataset.
            p1 (tuple): Bottom-left corner.
            p2 (tuple): Top-right corner.
            points (int): Number of data points.
            seed (int): Random seed.

        Returns:
            Data: The generated dataset.
        """
        data = []
        random.seed(seed)

        x_diff = p2[0] - p1[0]
        y_diff = p2[1] - p1[1]

        for _ in range(points):
            x_rand = random.random()
            y_rand = random.random()
            data.append(np.array([x_diff * x_rand + p1[0], y_diff * y_rand + p1[1]]))

        data = Data(data)
        data.save_data(output_file)
        return data

    def create_dataset_eigth_sphere(self, output_file: str = None, radius: float = 1, x_pos: bool = True, y_pos: bool = True, z_pos: bool = True, points: int = 1000, seed: int = 42, stream: bool = False) -> 'Data':
        """
        Creates a dataset on an eighth of a sphere.

        Args:
            output_file (str): File path to save the dataset.
            radius (float): Radius of the sphere.
            x_pos (bool): Use positive x.
            y_pos (bool): Use positive y.
            z_pos (bool): Use positive z.
            points (int): Number of data points.
            seed (int): Random seed.
            stream (bool): Whether to stream to file.

        Returns:
            Data: The generated dataset.
        """
        return self.stream_dataset_creator(output_file, self.fg.eigth_sphere_generator, seed, stream, radius, x_pos, y_pos, z_pos, points)

    def create_dataset_triangle(self, output_file: str = None, edges: list = [[0, 0], [1, 1], [2, 0]], points: int = 1000, seed: int = 42, stream: bool = False) -> 'Data':
        """
        Creates a dataset of points on a triangle.

        Args:
            output_file (str): File path to save the dataset.
            edges (list): Three vertices of the triangle.
            points (int): Number of data points.
            seed (int): Random seed.
            stream (bool): Whether to stream to file.

        Returns:
            Data: The generated dataset.
        """
        return self.stream_dataset_creator(output_file, self.fg.triangle_generator, seed, stream, edges, points)

    def create_dataset_strong_clusters(self, output_file: str = None, internal_std: float = 1, external_std: float = 10, mean: list = [0, 0], clusters: int = 10, points: int = 1000, seed: int = 42, stream: bool = False) -> 'Data':
        """
        Creates a clustered dataset with multiple clusters.

        Args:
            output_file (str): File path to save the dataset.
            internal_std (float): Standard deviation inside a cluster.
            external_std (float): Spread of cluster centers.
            mean (list): Mean location for generating cluster centers.
            clusters (int): Number of clusters.
            points (int): Number of data points.
            seed (int): Random seed.
            stream (bool): Whether to stream to file.

        Returns:
            Data: The generated dataset.
        """
        data = []
        random.seed(seed)
        np_mean = np.array(mean)

        cluster_centers = [varied_point(np_mean, external_std) for _ in range(clusters)]

        if stream:
            self.fg.setGenerator(self.fg.strong_cluster_generator)
            self.fg.stream_save(output_file, internal_std, cluster_centers, points)
            data = Data(output_file, stream=True)
        else:
            for p in self.fg.strong_cluster_generator(internal_std, cluster_centers, points):
                data.append(p)
            data = Data(data)
            data.save_data(output_file)

        return data

    def rotate_into_dimention(self, data: 'Data', higher_dim: int = 3, seed: int = 42) -> 'Data':
        """
        Rotates dataset into a higher dimensional space using random rotations.

        Args:
            data (Data): The dataset to rotate.
            higher_dim (int): Dimension to rotate into.
            seed (int): Random seed.

        Returns:
            Data: The rotated dataset.
        """
        rotation_matrix = np.identity(higher_dim)
        if seed != -1:
            random.seed(seed)

        for x1 in range(higher_dim - 1):
            for x2 in range(x1 + 1, higher_dim):
                angle = 2 * np.pi * random.random()
                rot = np.identity(higher_dim)
                rot[x1, x1] = np.cos(angle)
                rot[x2, x2] = np.cos(angle)
                rot[x1, x2] = np.sin(angle)
                rot[x2, x1] = -np.sin(angle)
                rotation_matrix = np.matmul(rotation_matrix, rot)

        data.data = list(data.data)
        for i in range(len(data)):
            extended = np.zeros(higher_dim)
            extended[:len(data[i])] = data[i]
            data[i] = np.matmul(rotation_matrix, extended)

        data.data = np.array(data.data)
        return data

    def create_dataset_spiral(self, output_file: str = None, radius: float = 1, center: list = [0, 0], rotations: int = 3, height: float = 10, points: int = 1000, seed: int = 42, stream: bool = False) -> 'Data':
        """
        Creates a 3D spiral dataset.

        Args:
            output_file (str): File path to save the dataset.
            radius (float): Radius of the spiral.
            center (list): Center offset.
            rotations (int): Number of rotations.
            height (float): Height of the spiral.
            points (int): Number of data points.
            seed (int): Random seed.
            stream (bool): Whether to stream to file.

        Returns:
            Data: The generated dataset.
        """
        return self.stream_dataset_creator(output_file, self.fg.spiral_generator, seed, stream, radius, center, rotations, height, points)

if __name__ == '__main__':
	creator = DataCreator()

	print("Making line...")
	line_points = creator.create_dataset_line(output_file="../inputoutput/data/line.json", start=0, end=3, seed=time.time())

	print("Making square edge...")
	edge_points = creator.create_dataset_square_edge(output_file="../inputoutput/data/square_edge.json", seed=time.time())

	print("Making square fill...")
	square_points = creator.create_dataset_square_fill(output_file="../inputoutput/data/square_fill.json", seed=time.time())

	print("Making eigth sphere fill...")
	creator.create_dataset_eigth_sphere(output_file="../inputoutput/data/eigth_sphere.json", seed=time.time())

	print("Making strong clusters...")
	creator.create_dataset_strong_clusters(output_file="../inputoutput/data/strong_clusters.json", seed=time.time())

	print("Making spiral...")
	creator.create_dataset_spiral(output_file="../inputoutput/data/spiral.json", seed=time.time())

	print("Making triangle...")
	creator.create_dataset_triangle(output_file="../inputoutput/data/triangle.json", seed=time.time())

	print("Line in 3D")
	creator.rotate_into_dimention(line_points, seed=time.time()).save_data("../inputoutput/data/3d_line.json")

	print("Square Fill in 3D")
	square_3d = creator.rotate_into_dimention(square_points, seed=time.time()).save_data("../inputoutput/data/3d_square.json")

	print("Square Edge in 10D")
	creator.rotate_into_dimention(edge_points, higher_dim=10, seed=time.time()).save_data("../inputoutput/data/10d_square_edge.json")

	# square_3d.plot()

	"""
	print("Large Line")
	creator.create_dataset_line(output_file="../inputoutput/data/large_line.json", points=1000000, seed=time.time(), stream=True)

	print("Large Single Cluster")
	creator.create_dataset_strong_clusters(output_file="../inputoutput/data/large_single_cluster.json", clusters=1, points=1000000, seed=time.time(), stream=True)

	print("Large Many Clusters")
	creator.create_dataset_strong_clusters(output_file="../inputoutput/data/large_many_clusters.json", internal_std=100, external_std=1000, mean=[0, 0, 0], clusters=10000, points=1000000, seed=time.time(), stream=True)

	print("Large eigth sphere")
	creator.create_dataset_eigth_sphere(output_file="../inputoutput/data/large_eigth_sphere.json", points=1000000, seed=time.time(), stream=True)

	print("Large spiral")
	creator.create_dataset_spiral(output_file="../inputoutput/data/large_spiral.json", points=1000000, seed=time.time(), stream=True)
	print("Large triangle")
	creator.create_dataset_triangle(output_file="../inputoutput/data/large_triangle.json", points=1000000, seed=time.time(), stream=True)
	# """

	# data = Data("large_line.json", stream=True)
	# for point in data:
	# 	print(point)

	# print("Making weak clusters...")
	# create_dataset_weak_clusters(output_file="weak_clusters.json", seed=time.time())
