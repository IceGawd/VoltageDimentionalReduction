import landmark
import problem

from typing import List
import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import csr_matrix

class Solver:
    """
    Solves for voltage distributions across a set of points in a resistance network.

    Given a problem with defined resistances and a set of landmarks with fixed voltages,
    this class computes the approximate voltages at all other points.

    Attributes:
        problem (Problem): The resistance network model.
    """

    def __init__(self, problem: problem.Problem):
        """
        Initializes the solver with a given problem.

        Args:
            problem (Problem): The problem instance defining the resistance matrix.
        """
        self.problem = problem

    def approximate_voltages(self, landmarks: List[landmark.Landmark], max_iters: int = 100) -> np.ndarray:
        """
        Computes approximate voltages for all points in the network given landmark constraints.

        Args:
            landmarks (List[Landmark]): List of Landmark instances with indices and voltage values.
            max_iters (int): Maximum number of iterations for the solver.

        Returns:
            np.ndarray: Array of voltages of size (n,) where n is the number of points.
        """
        n = self.problem.points.shape[0]
        R = self.problem.calcResistanceMatrix()  # (n+1)x(n+1) matrix
        R = R.tocsr() if not isinstance(R, csr_matrix) else R

        R_n = R[:-1, :-1]
        ground_column = R[:-1, -1].toarray().flatten()

        # Initialize voltage vector and fixed mask
        V = np.zeros(n)
        fixed_mask = np.zeros(n, dtype=bool)
        fixed_values = {}

        for landmark in landmarks:
            fixed_mask[landmark.index] = True
            fixed_values[landmark.index] = landmark.voltage
            V[landmark.index] = landmark.voltage

        # Identify free indices and adjust linear system
        free_indices = np.where(~fixed_mask)[0]
        A = R_n[free_indices][:, free_indices]

        b = ground_column[free_indices].copy()
        if any(fixed_mask):
            fixed_indices = np.where(fixed_mask)[0]
            fixed_vector = np.array([fixed_values[i] for i in fixed_indices])
            b -= R_n[free_indices][:, fixed_indices] @ fixed_vector

        # Solve the system
        V_free, _ = cg(A, b, maxiter=max_iters)
        V[free_indices] = V_free

        return V