import numpy as np
import matplotlib.pyplot as plt
from typing import List

class Visualization:
    """
    Visualization utilities for displaying voltage maps on 2D point sets.

    Primarily intended to visualize outputs from the Solver.
    """

    @staticmethod
    def plot_voltage_map(points: np.ndarray, voltages: np.ndarray, title: str = "Voltage Map") -> None:
        """
        Plots a 2D scatter plot where point colors correspond to voltage values.

        Args:
            points (np.ndarray): n x 2 array of 2D coordinates.
            voltages (np.ndarray): 1D array of voltages, same length as number of points.
            title (str): Title for the plot.
        """
        if points.shape[1] != 2:
            raise ValueError("Visualization only supports 2D data.")

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(points[:, 0], points[:, 1], c=voltages, cmap='viridis', s=30, edgecolor='k')
        plt.colorbar(scatter, label='Voltage')
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_voltage_maps(points: np.ndarray, voltage_maps: List[np.ndarray], titles: List[str]) -> None:
        """
        Plots multiple voltage maps side-by-side for comparison.

        Args:
            points (np.ndarray): n x 2 array of 2D coordinates.
            voltage_maps (List[np.ndarray]): List of 1D voltage arrays.
            titles (List[str]): Titles for each subplot.
        """
        if points.shape[1] != 2:
            raise ValueError("Visualization only supports 2D data.")
        if len(voltage_maps) != len(titles):
            raise ValueError("Length of voltage_maps and titles must match.")

        n = len(voltage_maps)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        for i in range(n):
            sc = axes[i].scatter(points[:, 0], points[:, 1], c=voltage_maps[i], cmap='viridis', s=30, edgecolor='k')
            axes[i].set_title(titles[i])
            axes[i].set_xlabel("X")
            axes[i].set_ylabel("Y")
            axes[i].grid(True)
            plt.colorbar(sc, ax=axes[i], label='Voltage')

        plt.tight_layout()
        plt.show()