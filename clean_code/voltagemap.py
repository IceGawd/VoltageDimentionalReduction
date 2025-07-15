import landmark
import solver
import problem

import numpy as np
from typing import List, Dict

class VoltageMap:
    """
    A container class that manages and processes voltage solutions for multiple landmarks.

    This class represents a collection of voltage solutions (voltage maps), where each map
    corresponds to the solution obtained by applying a Solver to a Problem with a specific
    Landmark. It provides functionality for storing, retrieving, and manipulating these
    voltage maps.

    Attributes:
        entries (List[Tuple[Landmark, np.ndarray, float]]): List of tuples containing:
            - landmark: The Landmark instance
            - voltages: The computed voltage map (numpy array)
            - advantage: The norm of the voltage map
        shape (tuple): The shape of the voltage maps stored in this container

    Example:
        >>> voltage_map = VoltageMap()
        >>> voltage_map.add_solution(landmark_obj, computed_voltages)
        >>> voltages, advantage = voltage_map.get_solution(landmark_index=0)
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
        Adds a voltage map corresponding to a specific landmark to the collection.

        This method stores a new voltage map along with its corresponding landmark
        and automatically computes its advantage (norm of the voltage map).
        If this is the first voltage map added, it also sets the shape attribute.

        Args:
            landmark_obj (Landmark): The landmark instance used to generate the voltage map
            voltages (np.ndarray): The computed voltage map for that landmark

        Note:
            The advantage is computed as the Euclidean norm of the voltage map
            and is stored alongside the solution.
        """
        # Calculate advantage as the norm of the voltage map
        advantage = np.linalg.norm(voltages)
        
        self.entries.append({
            "landmark": landmark_obj,
            "voltages": voltages,
            "advantage": advantage  # Add advantage to the entry
        })


    def sort_by_advantage(self, quantity="advantage", reverse=True) -> None:
        """
        Sorts the entries by the specified quantity (default advantage).
        
        Args:
            quantity (str): The quantity to sort by. Defaults to 'advantage'.
            reverse (bool): If True, sort in descending order. Defaults to True.
            
        Raises:
            KeyError: If the specified quantity doesn't exist in any entry
        """
        # Check if the quantity exists in the entries
        if not self.entries:
            return
            
        if quantity not in self.entries[0]:
            raise KeyError(f"Cannot sort by '{quantity}', quantity not found in entries. "
                         f"Available quantities: {list(self.entries[0].keys())}")
            
        self.entries.sort(key=lambda x: x[quantity], reverse=reverse)

    def all_solutions(self) -> np.ndarray:
        """
        Retrieves all voltage maps as a stacked 2D array.

        This method combines all stored voltage maps into a single 2D array where
        each column represents the voltages for a specific landmark, and each row
        represents the voltages at a specific point across all landmarks.

        Returns:
            np.ndarray: A 2D array of shape (num_points, num_landmarks) where:
                - Each column represents a voltage map for a landmark
                - Each row represents voltages at a point across all landmarks

        Note:
            The returned array is transposed from the stored format for convenience,
            making it easier to work with point-wise operations.
        """
        V=np.stack([E['voltages'] for E in self.entries], axis=0)
        return V.T

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        """
        Makes VoltageMap iterable over its voltage maps.

        Initializes the iteration state and returns self as iterator.
        This allows the class to be used in for loops to iterate over
        the voltage maps.

        Returns:
            VoltageMap: self as iterator object
        """
        self._iter_idx = 0
        return self

    def __next__(self):
        """
        Implements the iterator protocol for voltage maps.

        Provides sequential access to the voltage maps stored in this container.
        Each call returns the next voltage map in the sequence until all maps
        have been returned.

        Returns:
            np.ndarray: The next voltage map in the sequence

        Raises:
            StopIteration: When there are no more voltage maps to iterate over
        """
        if self._iter_idx >= len(self.entries):
            raise StopIteration
        voltages = self.entries[self._iter_idx]['voltages']
        self._iter_idx += 1
        return voltages

    ##YF:Does this belong here?
    @staticmethod
    def from_problem_and_landmarks(problem: problem.Problem, landmarks: list[landmark.Landmark], solver_cls: solver.Solver) -> "VoltageMap":
        """
        Factory method to create a VoltageMap from a problem and set of landmarks.

        This method creates a new VoltageMap instance and populates it by solving
        the given problem for each provided landmark using the specified solver class.
        It's a convenient way to generate a complete voltage map collection in one step.

        Args:
            problem (Problem): The problem instance to solve
            landmarks (List[Landmark]): List of landmarks to use for solving
            solver_cls (Solver): The solver class to use for computing voltage maps

        Returns:
            VoltageMap: A new VoltageMap instance containing solutions for all landmarks

        Example:
            >>> landmarks = [Landmark(i) for i in range(10)]
            >>> voltage_map = VoltageMap.from_problem_and_landmarks(
            ...     problem=my_problem,
            ...     landmarks=landmarks,
            ...     solver_cls=MySolver
            ... )
        """
        voltage_map = VoltageMap()
        for lm in landmarks:
            solver_instance = solver_cls(problem, lm)
            voltages = solver_instance.approximate_voltages()
            voltage_map.add_solution(lm, voltages)
        return voltage_map