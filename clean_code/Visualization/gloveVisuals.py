import matplotlib.pyplot as plt
import numpy as np
from typing import List

from Visualization import visualHelpers

def plot_text(transformed_points, words, voltages, ax, image_size, landmarkSize, alpha_actual, remove_clutter):
	landmark_indicies = [landmark.index for landmark in voltages.get_all_landmarks()]
	drawn_xy = []
	size_sqrd = image_size ** 2
	for i in range(transformed_points.shape[0]):
		landmark = i in landmark_indicies
		color = 'white'
		fontsize = 8
		
		if landmark:
			color = 'yellow'
			fontsize *= landmarkSize
			
		x, y = transformed_points[i]

		draw = True

		if remove_clutter:
			for (x2, y2) in drawn_xy:
				if ((x2 - x) ** 2 + (y2 - y) ** 2 < size_sqrd):
					draw = False
					break

		if draw:
			if remove_clutter:
				drawn_xy.append((x, y))

			ax.text(transformed_points[i, 0], transformed_points[i, 1], str(words[i]),
					color=color, fontsize=fontsize, alpha=alpha_actual,
					ha='center', va='center')

def plot_voltage_words(voltages, words, transformation="mds", landmarkSize=3, alpha_actual=1, percent_size=0.02, remove_clutter=True, out_file=None):
	"""
	Visualizes voltage-based word embeddings in 2D using MDS or PCA.
	Each word is plotted as text at its transformed position.

	Args:
		voltages: VoltageMap object containing the voltage embedding.
		words: List of strings to display.
		transformation: Dimensionality reduction method ('mds' or 'pca').
		landmarkSize: Size of landmark points.
		alpha_actual: Transparency of the text.
		percent_size: Controls image size scaling.
		remove_clutter: (Unused, for compatibility).
		out_file: If given, saves figure to file.
	"""
	points = voltages.voltage_array()
	transformed_points = visualHelpers.transform(points, transformation)
	x_bound, y_bound, image_size = visualHelpers.compute_image_size(transformed_points, percent_size)
	fig, ax = visualHelpers.setup_figure(x_bound, y_bound, image_size, landmarkSize, "Voltage Map Words")

	plot_text(transformed_points, words, voltages, ax, image_size, landmarkSize, alpha_actual, remove_clutter)

	visualHelpers.standard_save_display(out_file)