import landmark
import problem
from Utilities import config, timer
import setofpoints

from typing import Union, List
import numpy as np
from scipy.linalg import solve
import pandas as pd
import argparse

class Solver:
	"""
	Solves for voltage distributions across a set of points in a resistance network.

	Given a problem with defined resistances and a set of landmarks with fixed voltages,
	this class computes the approximate voltages at all other points.

	Attributes:
		problem (Problem): The resistance network model.
	"""

	def __init__(self, this_problem: problem.Problem):
		"""
		Initializes the solver with a given problem.

		Args:
			problem (Problem): The problem instance defining the resistance matrix.
		"""
		self.problem = this_problem

	def compute_voltages(self, this_landmark: landmark.Landmark):
		"""
		Computes and returns the voltages for the given problem.
		See for explaination: https://github.com/IceGawd/VoltageDimentionalReduction/blob/main/highLevelDocs/VoltageCalculation.md

		Args:
		this_landmark landmark.Landmark: The landmark

                link to the pdf 
                
		Returns:
			voltages: the voltages solution ndarray
		"""
		
		weights = self.problem.getResistanceMatrix()
		n = weights.shape[0]

		ground = landmark.Landmark(n - 1, 0)	
		landmarks = [this_landmark, ground]
		
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
		
		self.voltages = self.voltages[:-1]

		return self.voltages

# Example usage
def main():
	n = config.params.get("n", 11)
	k = config.params.get("k", 2)
	r = config.params.get("r", 1.0)
	landmark_index = config.params.get("landmark", n // 2)

	points = np.array([[x] for x in np.arange(0, 1, 1.0 / n)])
	simple_set = setofpoints.SetOfPoints(points)
	simple_problem = problem.Problem(simple_set, k = k, r = r)
	simple_solver = Solver(this_problem = simple_problem)
	voltages = simple_solver.compute_voltages(landmark.Landmark(landmark_index, 1))

	# Sanity check: each voltage should be a weighted average of its neighbors + ground
	R = simple_problem.getResistanceMatrix()
	V = voltages.tolist() + [0]  # Add ground voltage (0)

	for i in range(len(points)):
		if i != landmark_index:
			neighbors = R[i]
			avg = 0
			for j, weight in enumerate(neighbors):
				if j != i:
					avg -= weight * V[j]
			assert abs(V[i] - avg) < 1e-5, "Node " + str(i) + " failed average check: " + str(V[i]) + " vs " + str(avg)

	print("Voltage solution passed Kirchhoff sanity check.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Voltage map solver test")
	parser.add_argument("--n", type=int, default=11, help="Number of points")
	parser.add_argument("--k", type=int, default=2, help="Number of neighbors")
	parser.add_argument("--r", type=float, default=1.0, help="Resistance to ground")
	parser.add_argument("--landmark", type=int, help="Index of landmark point")

	args = parser.parse_args()
	
	# Update the global config
	config.params["n"] = args.n
	config.params["k"] = args.k
	config.params["r"] = args.r
	config.params["landmark"] = args.landmark if args.landmark is not None else args.n // 2

	main()
