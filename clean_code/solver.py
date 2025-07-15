import landmark
import problem
from Utilities import config
import setofpoints

from typing import Union, List
import numpy as np
from scipy.linalg import solve
import pandas as pd


class Solver:
	"""
	Solves for voltage distributions in a resistance network using numerical methods.

	This class implements a solver for computing voltage distributions across a set of points
	in a resistance network. It uses a system of linear equations based on Kirchhoff's laws
	to determine the voltages at each point given:
	- A resistance network defined by a Problem instance
	- A set of landmarks with fixed voltages
	- A ground point (automatically added)

	The solver uses the following approach:
	1. Constructs a resistance matrix from the problem
	2. Separates constrained (landmark) and unconstrained nodes
	3. Solves the resulting linear system using scipy.linalg.solve

	Attributes:
		problem (Problem): The resistance network model containing point set and resistance parameters
		voltages (np.ndarray): The most recently computed voltage solution (available after solve)

		Example:
			>>> problem_instance = Problem(point_set, r=1.0)
			>>> solver = Solver(problem_instance)
			>>> voltages = solver.compute_voltages(landmark_obj)
	"""

	def __init__(self, problem: problem.Problem):
		"""
		Initializes the solver with a given problem.

		Args:
			problem (Problem): The problem instance defining the resistance matrix.
		"""
		self.problem = problem

	def compute_voltages(self, this_landmark: landmark.Landmark):
		"""
		Computes voltage distribution across all points in the network.

		This method solves for the voltages at each point in the resistance network given
		a landmark with a fixed voltage. The solution process involves:
		1. Getting the resistance matrix from the problem
		2. Adding a ground point (voltage = 0) as an additional landmark
		3. Separating nodes into constrained (landmarks) and unconstrained sets
		4. Solving the linear system Ax = b where:
		   - A is the resistance matrix for unconstrained nodes
		   - b is the voltage contribution from landmark nodes
		   - x gives the voltages at unconstrained nodes

		Args:
			this_landmark (landmark.Landmark): The landmark specifying a point with fixed voltage

		Returns:
			np.ndarray: Computed voltages for all points except ground (shape: n-1)
					   where n is the total number of points including ground

		Note:
			- A ground point is automatically added as the last point with voltage = 0
			- The returned voltages array excludes the ground point
			- The solution is stored in self.voltages for later access
		"""
		
		### yf: I think most of this logic should reside in calcresistancematrix. The only
		### logic that should be here is incorporating the (single) landmark.

		weights = self.problem.getResistanceMatrix()

		n = weights.shape[0]

		ground=landmark.Landmark(n - 1, 0)	
		landmarks=[this_landmark,ground]
		
		constrained_nodes =   [l.index for l in landmarks]
		unconstrained_nodes = [i for i in range(n) if i not in constrained_nodes]
		
		# I don't understand the lines from here to the #print
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
	config.params['r'] = 1.0
	config.params['c'] = 1.0

	points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
	weights = np.array([1, 1, 1, 1])
	point_set = setofpoints.SetOfPoints(points=points, weights=weights)

	problem_instance = problem.Problem(point_set, r=config.params['r'])
	solver_instance = Solver(problem_instance)

	landmarks = [landmark.Landmark(0, 5), landmark.Landmark(1, 10)]
	voltages = solver_instance.compute_voltages(landmarks)
	print("Computed Voltages:", voltages)

	# main needs to implement a test that passes/fails without human input.
		

if __name__ == "__main__":
	main()
