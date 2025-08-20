from __future__ import annotations
import numpy as np
import solver

class VoltageMap:
    """
    A class for managing multiple voltage solutions on a graph with specified
    landmarks and their associated voltages.

    Attributes
    ----------
    solver_cls : type[solver.Solver]
        The solver class used for voltage calculations.
    entries : list of dict
        Each entry contains:
            - 'landmark' (list[int]): Landmark vertices.
            - 'voltages' (np.ndarray): Voltage array corresponding to the landmarks.
    all_solutions : list of dict
        Each entry contains:
            - 'landmark' (list[int]): Landmark vertices.
            - 'voltages' (list[np.ndarray]): List of voltage arrays for the landmarks.
    voltage_array : np.ndarray
        An array containing all voltage solutions.
    _iter_idx : int
        Internal index for iteration.
    """

    def __init__(self, solver_cls: type[solver.Solver]) -> None:
        """
        Initialize the VoltageMap with a solver class.

        Parameters
        ----------
        solver_cls : type[solver.Solver]
            The solver class used for voltage calculations.
        """
        self.solver_cls = solver_cls
        self.entries: list[dict] = []
        self.all_solutions: list[dict] = []
        self.voltage_array: np.ndarray | None = None
        self._iter_idx = 0

    def add_entry(self, landmark: list[int], voltages: np.ndarray) -> None:
        """
        Add a new landmark and its associated voltages.

        Parameters
        ----------
        landmark : list[int]
            The landmark vertices.
        voltages : np.ndarray
            The voltage array corresponding to the landmarks.
        """
        self.entries.append({"landmark": landmark, "voltages": voltages})

    def add_all_solutions(self, landmark: list[int], voltages: list[np.ndarray]) -> None:
        """
        Add multiple voltage solutions for a given landmark.

        Parameters
        ----------
        landmark : list[int]
            The landmark vertices.
        voltages : list[np.ndarray]
            List of voltage arrays corresponding to the landmarks.
        """
        self.all_solutions.append({"landmark": landmark, "voltages": voltages})

    def sort_entries(self) -> None:
        """
        Sort entries based on landmarks.
        """
        self.entries.sort(key=lambda e: e["landmark"])

    def get_voltage_array(self) -> np.ndarray:
        """
        Get the voltage array for all entries.

        Returns
        -------
        np.ndarray
            The stacked voltage arrays for all entries.
        """
        if self.voltage_array is None:
            self.voltage_array = np.array([entry["voltages"] for entry in self.entries])
        return self.voltage_array

    def get_all_landmarks(self) -> list[list[int]]:
        """
        Get all landmarks in the entries.

        Returns
        -------
        list of list[int]
            A list of all landmark vertex lists.
        """
        return [entry["landmark"] for entry in self.entries]

    def __iter__(self) -> VoltageMap:
        """
        Return an iterator over the entries.

        Returns
        -------
        VoltageMap
            The VoltageMap instance itself.
        """
        self._iter_idx = 0
        return self

    def __next__(self) -> np.ndarray:
        """
        Return the next voltage array in the iteration.

        Returns
        -------
        np.ndarray
            The next voltage array.

        Raises
        ------
        StopIteration
            If the iteration is complete.
        """
        if self._iter_idx < len(self.entries):
            voltages = self.entries[self._iter_idx]["voltages"]
            self._iter_idx += 1
            return voltages
        else:
            raise StopIteration
