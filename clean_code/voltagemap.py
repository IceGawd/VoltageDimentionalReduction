"""
Voltage Map Collection for Dimensionality Reduction

This module manages collections of voltage solutions computed from multiple landmarks
in an electrical network. It provides functionality to store, organize, and analyze
voltage distributions across different landmark configurations.

The voltage maps are used in the dimensionality reduction process, where each map
represents how electrical potential spreads from a specific landmark through the
network of points.

Example:
	>>> from voltagemap import VoltageMap
	>>> vmap = VoltageMap()
	>>> # Add solutions from multiple landmarks
	>>> vmap.add_solution(landmark1, voltages1)
	>>> vmap.add_solution(landmark2, voltages2)
	>>> # Get combined voltage array for dimensionality reduction
	>>> V = vmap.voltage_array()
"""

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

		Creates a new VoltageMap instance with no entries. Solutions can be
		added later using add_solution() or created from a problem using
		from_problem_and_landmarks().
		"""
		self.entries: List[Dict] = []  # Stores (landmark, voltages, advantage)

	def set_advantages(self, advantage: float, quantity: str = "advantage") -> None:
		"""
		Set a quality metric value for all voltage solutions.

		Updates the specified quality metric (default: 'advantage') for all
		entries in the voltage map. This can be used to score or rank different
		solutions based on their effectiveness.

		Args:
			advantage (float): The value to set for the quality metric.
			quantity (str, optional): Name of the quality metric to set.
				Defaults to "advantage".

		Example:
			>>> vmap = VoltageMap()
			>>> # Add some solutions...
			>>> vmap.set_advantages(0.5)  # Set default advantage
			>>> vmap.set_advantages(0.8, "quality_score")  # Set custom metric
		"""
		for i in range(len(self.entries)):
			self.entries[i][quantity] = advantage
			
	def add_solution(self, landmark_obj: landmark.Landmark, voltages: np.ndarray) -> None:
		"""
		Adds a voltage map corresponding to a specific landmark.

		Args:
			landmark_obj (Landmark): The landmark used in the problem.
			voltages (np.ndarray): The computed voltage map for that landmark.
		"""
		self.entries.append({
			"landmark":landmark_obj, 
			"voltages":voltages})


	def sort_by_advantage(self, quantity: str = "advantage", reverse: bool = True) -> None:
		"""
		Sort voltage solutions by their quality metric.

		Sorts the entries in the voltage map based on the specified quality
		metric. This is useful for ranking solutions or selecting the best
		landmarks based on some criterion.

		Args:
			quantity (str, optional): The metric to sort by. Defaults to "advantage".
			reverse (bool, optional): If True, sort in descending order
				(highest value first). If False, sort in ascending order.
				Defaults to True.

		Note:
			The specified quantity must exist in all entries. If an entry
			is missing the quantity, a KeyError will be raised.
		"""
		self.entries.sort(key=lambda x: x[quantity], reverse=reverse)

	def all_solutions(self) -> np.ndarray:
		"""
		Get all voltage solutions as a transposed 2D array.

		Combines all voltage solutions into a single array where each row
		represents a point and each column represents a landmark's voltage
		distribution.

		Returns:
			np.ndarray: 2D array of shape (num_points, num_landmarks) where:
				- Each column is a voltage solution from one landmark
				- Each row contains all landmark voltages for one point

		Example:
			>>> vmap = VoltageMap()
			>>> # Add solutions for 3 landmarks on 100 points...
			>>> V = vmap.all_solutions()  # Shape: (100, 3)

		Note:
			This method transposes the stacked solutions to match the
			expected format for dimensionality reduction algorithms.
		"""
		V=np.stack([E['voltages'] for E in self.entries], axis=0)
		return V.T

	def voltage_array(self) -> np.ndarray:
		"""
		Returns an (N x L) array of voltages, where N is the number of data points
		and L is the number of landmarks.

		Each column corresponds to the voltage map from one landmark.
		"""
		if not self.entries:
			raise ValueError("VoltageMap has no entries.")
		
		return np.column_stack([entry['voltages'] for entry in self.entries])

	def __len__(self) -> int:
		return len(self.entries)

	def __iter__(self):
		"""
		Returns an iterator over the voltage maps for use in for-loops.
		"""
		self._iter_idx = 0
		return self

	def __next__(self):
		"""
		Retrieves the next voltage map in an iteration.

		Returns:
			np.ndarray: The next voltage map.

		Raises:
			StopIteration: If the end of the map is reached.
		"""
		if self._iter_idx >= len(self.entries):
			raise StopIteration
		voltages = self.entries[self._iter_idx][1]
		self._iter_idx += 1
		return voltages

	##YF:Does this belong here?
	@staticmethod
	def from_problem_and_landmarks(
		problem: problem.Problem,
		landmarks: List[landmark.Landmark],
		solver_cls: type[solver.Solver]
	) -> "VoltageMap":
		"""
		Create a VoltageMap by solving a problem for multiple landmarks.

		This factory method automates the process of:
		1. Creating a solver for the given problem
		2. Computing voltage solutions for each landmark
		3. Collecting all solutions into a VoltageMap

		Args:
			problem (Problem): The resistance network problem to solve.
				Contains network structure and parameters.
			landmarks (List[Landmark]): List of landmarks to use as voltage
				sources. Each landmark specifies a point and voltage.
			solver_cls (type[Solver]): The Solver class to use for computing
				voltage distributions. Must implement approximate_voltages().

		Returns:
			VoltageMap: A new VoltageMap instance containing voltage solutions
				for all specified landmarks.

		Example:
			>>> prob = Problem(points, k=10)
			>>> landmarks = [Landmark(0, 1.0), Landmark(5, 1.0)]
			>>> vmap = VoltageMap.from_problem_and_landmarks(
			...     prob, landmarks, Solver
			... )

		Note:
			The solver_cls must be compatible with the problem and landmark
			types. It should implement the expected interface for voltage
			computation.
		"""
		voltage_map = VoltageMap()
		for lm in landmarks:
			solver_instance = solver_cls(problem, lm)
			voltages = solver_instance.approximate_voltages()
			voltage_map.add_solution(lm, voltages)
		return voltage_map