"""
Landmark Module

This module defines the Landmark class used in voltage-based dimensionality reduction.
Landmarks represent points in the dataset where voltages are applied to create
electrical potentials for dimensionality reduction.

Example:
    >>> from landmark import Landmark
    >>> # Create a landmark at index 5 with voltage 2.0
    >>> lm = Landmark(5, 2.0)
    >>> print(f"Voltage {lm.voltage}V at index {lm.index}")
"""

import numpy as np
from typing import Union, Optional, List, Any, Tuple, Callable, Dict

class Landmark:
    """
    Represents a location in the dataset where a voltage will be applied.

    A Landmark is a key component in voltage-based dimensionality reduction,
    defining where electrical potentials are applied in the resistance network
    model of the dataset.

    Attributes:
        index (int): Index identifying the point in the dataset where voltage is applied.
            Can refer to either an individual datapoint or a partition center.
        voltage (float): The voltage value to be applied at this landmark point.
            Default is 1.0V.
    """

    def __init__(self, index: int, voltage: float = 1.0) -> None:
        """
        Initialize a new Landmark instance.

        Creates a landmark point that will have a specified voltage in the
        resistance network model. The landmark is identified by its index
        in the dataset.

        Args:
            index (int): Index of the datapoint or partition center where
                voltage will be applied. Must be a valid index in the dataset.
            voltage (float, optional): Voltage value to apply at this landmark.
                Defaults to 1.0V. Must be a finite number.

        Example:
            >>> # Create landmark at first point with default voltage
            >>> lm1 = Landmark(0)
            >>> # Create landmark at point 10 with 2.5V
            >>> lm2 = Landmark(10, 2.5)

        """
        self.index = index
        self.voltage = voltage

	