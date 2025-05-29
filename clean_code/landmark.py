import numpy as np

class Landmark:
    """
    Represents a location in the dataset where a voltage will be applied.

    The `index` can refer either to an individual datapoint or a partition center.
    """

    def __init__(self, index: int, voltage: float) -> None:
        """
        Initializes a Landmark.

        Args:
            index (int): Index of the datapoint or partition center.
            voltage (float): Voltage to be applied at the specified index.
        """
        self.index = index
        self.voltage = voltage

    @staticmethod
    def createLandmarkClosestTo(
        data: List[Any],
        point: Any,
        voltage: float,
        distanceFn: Optional[object] = None,
        ignore: List[int] = []
    ) -> "Landmark":
        """
        Creates a Landmark at the index of the datapoint in `data` closest to `point`.

        Args:
            data (List[Any]): The dataset to search over.
            point (Any): The reference point to find the closest datapoint to.
            voltage (float): The voltage to assign to the resulting Landmark.
            ignore (List[int], optional): List of indices to skip during the search. Defaults to empty list.

        Returns:
            Landmark: A Landmark instance corresponding to the closest datapoint.
        """

        most_central_index = 0
        mindist = np.linalg.norm(data[0] - point)

        for index in range(1, len(data)):
            if index in ignore:
                continue

            dist = np.linalg.norm(data[index] - point)
            if dist < mindist:
                most_central_index = index
                mindist = dist

        return Landmark(most_central_index, voltage)