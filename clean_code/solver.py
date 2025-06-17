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

	def compute_voltages(self, landmarks: List["Landmark"], k: int = 10, universalGround: bool = True):
		"""
		Computes and returns the voltages for the given problem

		Args:
			landmarks (List["Landmark"]): The landmarks to consider when computing voltages

		Returns:
			voltages (List[float]): The voltages corresponding to each point in set of points
		"""

		weights = self.problem.calcResistanceMatrix(k, universalGround)
		n = weights.shape[0]

		if (universalGround):
			landmarks.append(landmark.Landmark(n - 1, 0))
		
		constrained_nodes =   [l.index for l in landmarks]
		unconstrained_nodes = [i for i in range(n) if i not in constrained_nodes]
		
		b = np.zeros(n)
		for lm in landmarks:
			for y in range(0, n):
				b[y] -= lm.voltage * weights[y][lm.index]
		
		A_unconstrained = weights[np.ix_(unconstrained_nodes, unconstrained_nodes)]
		b_unconstrained = b[unconstrained_nodes]

		# print(A_unconstrained, b_unconstrained)

		v_unconstrained = solve(A_unconstrained, b_unconstrained)

		self.voltages = np.zeros(n)

		for lm in landmarks:
			self.voltages[lm.index] = lm.voltage

		self.voltages[unconstrained_nodes] = v_unconstrained
		
		if (universalGround):
			self.voltages = self.voltages[:-1]

		return self.voltages

if __name__ == "__main__":
	# Example usage
	import setofpoints
	import config
	import pandas as pd

	config.params['r'] = 1.0
	config.params['c'] = 1.0

	points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
	weights = np.array([1, 1, 1, 1])
	point_set = setofpoints.SetOfPoints(points=points, weights=weights)

	problem_instance = problem.Problem(point_set, r=config.params['r'], c=config.params['c'])
	solver_instance = Solver(problem_instance)

	landmarks = [landmark.Landmark(0, 5), landmark.Landmark(1, 10)]
	voltages = solver_instance.compute_voltages(landmarks)
	print("Computed Voltages:", voltages)
