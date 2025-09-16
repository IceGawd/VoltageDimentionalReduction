import landmark
import solver
import problem

import numpy as np
from typing import List, Dict, Iterator, Optional

class VoltageMap:
	"""
	Collection of voltage solutions for multiple landmarks in a resistance network.

	This class manages voltage distributions computed from different landmark points,
	storing both the landmark information and corresponding voltage solutions.
	It provides methods to manipulate, sort, and analyze these solutions for
	dimensionality reduction purposes.

	Attributes:
		entries (List[Dict]): List of dictionaries, each containing:
			- landmark: Landmark object defining voltage source
			- voltages: np.ndarray of computed voltages
			- advantage: Optional quality metric for the solution

	Note:
		- Solutions can be sorted by advantage or other metrics
		- Supports iteration over voltage solutions
		- Provides methods to combine solutions into arrays
		- Handles construction from problems and landmark sets
	"""

	def __init__(self) -> None:
		"""
		Initialize an empty voltage map collection.
		"""
		self.entries: List[Dict] = []  # Stores dicts: {"landmark", "voltages", "advantage"}

	def set_advantages(self, advantage: float, quantity: str = "advantage") -> None:
		"""
		Set a quality metric value for all voltage solutions.
		"""
		for i in range(len(self.entries)):
			self.entries[i][quantity] = advantage
			
	def add_solution(self, landmark_obj: landmark.Landmark, voltages: np.ndarray) -> None:
		"""
		Adds a voltage map corresponding to a specific landmark.
		"""
		self.entries.append({
			"landmark": landmark_obj, 
			"voltages": voltages
		})

	def sort_by_advantage(self, quantity: str = "advantage", reverse: bool = True) -> None:
		"""
		Sort voltage solutions by their quality metric.
		"""
		self.entries.sort(key=lambda x: x[quantity], reverse=reverse)

	def all_solutions(self) -> np.ndarray:
		"""
		Get all voltage solutions as a transposed 2D array (num_points x num_landmarks).
		"""
		V = np.stack([E["voltages"] for E in self.entries], axis=0)
		return V.T

	def voltage_array(self) -> np.ndarray:
		"""
		Returns an (N x L) array of voltages, where N = number of data points
		and L = number of landmarks.
		"""
		if not self.entries:
			raise ValueError("VoltageMap has no entries.")
		return np.column_stack([entry["voltages"] for entry in self.entries])

	def get_all_landmarks(self) -> List[landmark.Landmark]:
		"""
		Returns a list of all Landmark objects in the VoltageMap.
		"""
		return [entry["landmark"] for entry in self.entries]

	def __len__(self) -> int:
		return len(self.entries)

	def __iter__(self) -> Iterator[np.ndarray]:
		"""
		Returns an iterator over the voltage maps for use in for-loops.
		"""
		self._iter_idx = 0
		return self

	def __next__(self) -> np.ndarray:
		"""
		Retrieves the next voltage map in an iteration.
		"""
		if self._iter_idx >= len(self.entries):
			raise StopIteration
		voltages = self.entries[self._iter_idx]["voltages"]
		self._iter_idx += 1
		return voltages

	@staticmethod
	def from_problem_and_landmarks(
		problem: problem.Problem,
		landmarks: List[landmark.Landmark],
		solver_cls: type[solver.Solver]
	) -> "VoltageMap":
		"""
		Create a VoltageMap by solving a problem for multiple landmarks.
		"""
		voltage_map = VoltageMap()
		for lm in landmarks:
			solver_instance = solver_cls(problem, lm)
			voltages = solver_instance.approximate_voltages()
			voltage_map.add_solution(lm, voltages)
		return voltage_map
