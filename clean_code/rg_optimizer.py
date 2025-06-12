"""
rg_optimizer.py

Module for optimizing the ground resistance parameter r in the Problem class.

This module provides a function to search for the best value of r by minimizing a user-specified loss function (e.g., validation error, voltage error, etc.).
"""

import numpy as np
from typing import Callable, Any
import networkx as nx

import solver

class rg_optimizer:
	"""
	Class for optimizing the ground resistance parameter r in a Problem instance.

	Attributes:
		problem (Problem): The Problem instance to optimize.
	"""
	def __init__(self, problem):
		"""
		Initializes the optimizer with a Problem instance.

		Args:
			problem (Problem): An instance of the Problem class.
		"""
		self.problem = problem


