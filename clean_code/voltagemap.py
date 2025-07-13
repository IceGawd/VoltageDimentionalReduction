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
        self.entries: list = []  # (landmark, voltages, advantage)

    def set_advantages(self, advantage: float, quantity="advantage") -> None:
        """
        Sets the advantage for all entries in the map to a specific value.

        Args:
            advantage (float): The advantage value to set for all entries.
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


    def sort_by_advantage(self, quantity="advantage",reverse=True) -> None:
        """
        Sorts the entries by the quantity (default advantage).
        """
        self.entries.sort(key=lambda x: x[quantity], reverse=reverse)

    def all_solutions(self) -> np.ndarray:
        """
        Retrieves all voltage maps as a stacked 2D array (landmarks x points).

        Returns:
            np.ndarray: 2D array of shape (num_landmarks, num_points)
        """
        V=np.stack([E['voltages'] for E in self.entries], axis=0)
        return V.T

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
    def from_problem_and_landmarks(problem: problem.Problem, landmarks: list[landmark.Landmark], solver_cls: solver.Solver) -> "VoltageMap":
        """
        Constructs a VoltageMap by solving the Problem for each landmark.

        Args:
            problem: An instance of a Problem class.
            landmarks (List[Landmark]): List of Landmark instances.
            solver_cls: A Solver class that takes a problem and a landmark.

        Returns:
            VoltageMap: A populated VoltageMap instance.
        """
        voltage_map = VoltageMap()
        for lm in landmarks:
            solver_instance = solver_cls(problem, lm)
            voltages = solver_instance.approximate_voltages()
            voltage_map.add_solution(lm, voltages)
        return voltage_map