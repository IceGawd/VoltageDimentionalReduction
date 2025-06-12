import landmark
import solver
import problem

import numpy as np
from typing import List, Dict

class VoltageMap:
	"""
	Represents a collection of voltage solutions (voltage maps), one for each landmark.

	Each voltage map corresponds to the solution from applying a Solver to a Problem with a specific Landmark.
	"""

	def __init__(self) -> None:
		"""
		Initializes an empty Map.
		"""
		self.voltage_maps: List[np.ndarray] = []		# Maps landmark index to voltage array
		self.landmarks: List[landmark.Landmark] = []	# Ordered list of landmarks
		self.shape: tuple = ()

	def add_solution(self, landmark_index: landmark.Landmark, voltages: np.ndarray) -> None:
		"""
		Adds a voltage map corresponding to a specific landmark.

		Args:
			landmark_index (Landmark): The landmark used in the problem.
			voltages (np.ndarray): The computed voltage map for that landmark.
		"""
		self.voltage_maps.append(voltages)
		self.landmarks.append(landmark_index)
		if not self.shape:
			self.shape = voltages.shape

	def get_solution(self, landmark_index: int) -> np.ndarray:
		"""
		Retrieves the voltage map for a specific landmark.

		Args:
			landmark_index (int): Index of the desired landmark.

		Returns:
			np.ndarray: The voltage map.
		"""
		return self.voltage_maps[landmark_index]

	def all_solutions(self) -> np.ndarray:
		"""
		Retrieves all voltage maps as a stacked 2D array (landmarks x points).

		Returns:
			np.ndarray: 2D array of shape (num_landmarks, num_points)
		"""
		return np.stack([self.voltage_maps[lm] for lm in self.landmarks], axis=0)

	def __len__(self) -> int:
		return len(self.voltage_maps)

	def __iter__(self):
		"""
		Returns an iterator over the dataset for use in for-loops.

		Returns:
			Iterator: An iterator over the dataset.
		"""
		self.index = -1

		return self

	def __next__(self):
		"""
		Retrieves the next voltage list in an iteration.

		Returns:
			List[np.ndarray]: The next voltage list.

		Raises:
			StopIteration: If the end of the map is reached.
		"""
		if (self.index + 1 == len(self.voltage_maps)):
			raise StopIteration
		else:
			self.index += 1
			return self.voltage_maps[self.index]

	@staticmethod
	def from_problem_and_landmarks(problem: problem.Problem, landmarks: List[landmark.Landmark], solver_cls: solver.Solver) -> "Map":
		"""
		Constructs a Map by solving the Problem for each landmark.

		Args:
			problem: An instance of a Problem class.
			landmarks (List[Landmark]): List of Landmark instances.
			solver_cls: A Solver class that takes a problem and a landmark.

		Returns:
			Map: A populated Map instance.
		"""
		voltage_map = Map()
		for landmark in landmarks:
			solver = solver_cls(problem, landmark)
			voltages = solver.approximate_voltages()
			voltage_map.add_solution(landmark.index, voltages)
		return voltage_map