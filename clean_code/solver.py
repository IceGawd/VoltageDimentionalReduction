import landmark
import problem

from typing import List
import numpy as np
from scipy.linalg import solve

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

	def compute_voltages(self, k: int = 10):
		"""
		Computes and returns the voltages for the given problem

		Returns:
			voltages (List[float]): The voltages corresponding to each point in set of points
		"""

		weights = self.problem.calcResistanceMatrix(k)

		n = weights.shape[0]
		
		constrained_nodes =   [l.index for l in self.problem.landmarks]
		unconstrained_nodes = [i for i in range(n) if i not in constrained_nodes]
		
		b = np.zeros(n)
		for landmark in self.problem.landmarks:
			for y in range(0, n):
				b[y] += landmark.voltage * weights[y][landmark.index]
		
		A_unconstrained = np.identity(len(unconstrained_nodes)) - weights[np.ix_(unconstrained_nodes, unconstrained_nodes)]
		b_unconstrained = b[unconstrained_nodes]
		v_unconstrained = solve(A_unconstrained, b_unconstrained)

		self.voltages = np.zeros(n)

		for landmark in self.problem.landmarks:
			self.voltages[landmark.index] = landmark.voltage

		self.voltages[unconstrained_nodes] = v_unconstrained
		
		if (self.problem.universalGround):
			self.voltages = self.voltages[:-1]

		return self.voltages
