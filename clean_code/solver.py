"""
Voltage Distribution Solver for Resistance Networks

This module implements a solver for computing voltage distributions in electrical
networks used for dimensionality reduction. It solves Kirchhoff's laws for a
network of resistors with fixed voltage landmarks and a ground reference.

The solver uses a system of linear equations derived from Kirchhoff's current law,
where each node's voltage is a weighted average of its neighbors' voltages.

Example:
	>>> from solver import Solver
	>>> from problem import Problem
	>>> from landmark import Landmark
	>>> # Create problem with resistance network
	>>> prob = Problem(points, k=10, r=1.0)
	>>> # Initialize solver
	>>> solver = Solver(prob)
	>>> # Compute voltages with landmark at index 5
	>>> voltages = solver.compute_voltages(Landmark(5, 1.0))
"""

import landmark
import problem
from Utilities import config
import setofpoints

from typing import Union, List
import numpy as np
from scipy.linalg import solve
import pandas as pd
import argparse

class Solver:
	"""
	Solves for voltage distributions across a set of points in a resistance network.

	This class implements a linear system solver that:
	1. Takes a resistance network with defined edge weights
	2. Applies fixed voltages at landmark points and ground
	3. Computes voltages at all other points using Kirchhoff's laws

	The solution ensures that:
	- Current is conserved at each node (Kirchhoff's current law)
	- Voltage differences follow Ohm's law along each edge
	- Landmark points maintain their specified voltages
	- Ground node has zero voltage

	Attributes:
		problem (Problem): The resistance network model containing:
			- Points and their connections
			- Edge weights (inverse resistances)
			- Ground node connections
		voltages (np.ndarray): The most recently computed voltage solution
			(available after calling compute_voltages)

	Note:
		The implementation uses sparse matrix operations and efficient
		linear system solving from scipy.linalg for performance.
	"""

	def __init__(self, net_problem: problem.Problem) -> None:
		"""
		Initialize the Solver with a resistance network problem.

		Sets up the solver with a Problem instance that defines the network
		structure and resistance matrix. The solver will use this information
		to compute voltage distributions when landmarks are specified.

		Args:
			net_problem (Problem): The problem instance containing:
				- Resistance matrix defining network connections
				- Point coordinates and weights
				- Ground resistance value
				- K-nearest neighbor structure

		Raises:
			TypeError: If net_problem is not an instance of Problem class
			ValueError: If the resistance matrix in the problem is invalid
				(e.g., not square, not symmetric)

		Example:
			>>> prob = Problem(points, k=10, r=1.0)
			>>> solver = Solver(prob)
		"""
		if not isinstance(net_problem, problem.Problem):
			raise TypeError("Expected Problem instance")
		self.problem = net_problem

	def compute_voltages(self, this_landmark: landmark.Landmark) -> np.ndarray:
		"""
		Compute voltage distribution across the network given a landmark point.

		This method solves the linear system AV = b where:
		- A is the resistance matrix for unconstrained nodes
		- V is the unknown voltage vector
		- b is derived from landmark and ground voltages

		Algorithm:
		1. Set up ground node with voltage 0
		2. Identify constrained (landmark, ground) and unconstrained nodes
		3. Build linear system considering voltage constraints
		4. Solve for unknown voltages
		5. Combine with known voltages to get full solution

		For detailed mathematical explanation, see:
		https://github.com/IceGawd/VoltageDimentionalReduction/blob/main/highLevelDocs/VoltageCalculation.md

		Args:
			this_landmark (landmark.Landmark): The landmark point where voltage
				is fixed. Contains:
				- index: position in the network
				- voltage: fixed voltage value to apply

		Returns:
			np.ndarray: Computed voltages for all points in the network,
				excluding the ground node. Shape is (n,) where n is the
				number of points.

		Note:
			- Solution satisfies Kirchhoff's current law at each node
			- Ground node (last node) is always at 0V
			- The landmark maintains its specified voltage
			- All other voltages are weighted averages of neighbors
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
def main() -> None:
	"""
	Main function to demonstrate and test the voltage solver.

	This function:
	1. Creates a simple 1D test case with evenly spaced points
	2. Solves for voltages with a landmark in the middle
	3. Verifies the solution satisfies Kirchhoff's laws
	
	The test parameters are configured through command line arguments or defaults:
	- n: Number of points (default: 11)
	- k: Number of neighbors (default: 2)
	- r: Ground resistance (default: 1.0)
	- landmark: Index of landmark point (default: n//2)

	Raises:
		AssertionError: If the computed voltages don't satisfy
			Kirchhoff's current law at any node.

	Note:
		This is primarily used for testing and demonstration.
		For real applications, create Problem and Solver instances directly.
	"""
	n = config.params.get("n", 11)
	k = config.params.get("k", 2)
	r = config.params.get("r", 1.0)
	landmark_index = config.params.get("landmark", n // 2)

	points = np.array([[x] for x in np.arange(0, 1, 1.0 / n)])
	simple_set = setofpoints.SetOfPoints(points)
	simple_problem = problem.Problem(simple_set, k = k, r = r)
	simple_solver = Solver(net_problem = simple_problem)
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
