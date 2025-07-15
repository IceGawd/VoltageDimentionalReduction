
"""
Landmark representation for voltage application in a dataset.

This module defines the Landmark class, which is used to represent a location (by index)
in a dataset where a voltage will be applied. The index can refer to a datapoint or a partition center.
"""

import numpy as np
from typing import Union, Optional, List, Any, Tuple, Callable, Dict

class Landmark:
	"""
	Represents a location in the dataset where a voltage will be applied.

	The `index` can refer either to an individual datapoint or a partition center.
	"""

	def __init__(self, index: int, voltage:float =1.0) -> None:
		"""
		Initializes a Landmark.

		Args:
			index (int): Index of the datapoint or partition center.
			voltage (float): Voltage to be applied at the specified index.
		"""
		self.index = index
		self.voltage=voltage

	