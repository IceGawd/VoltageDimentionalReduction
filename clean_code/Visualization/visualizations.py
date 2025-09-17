""" A wrapper around mnistVisuals.py"""


import sys, os
sys.path.append(os.path.abspath("../clean_code/"))
from Utilities import config
from Utilities import set_params
set_params.set_params()

import pickle
import numpy as np

import visualHelpers
from mnistVisuals import plot_landmark_subset, plot_points_from_file


def main():
	set_params.set_params()

	# Load data from the pickle file
	save_data = config.params['save_data']
	print(f"Loading data from {save_data}")
	
	with open(save_data, 'rb') as f:
		data = pickle.load(f)
	
	# Extract data components
	voltages = data['voltage_map']
	centroids = data['centroids']
	label_counts = data['label_counts']
	centroid_others = data['centroid_others']
	points = voltages.voltage_array()

	indices = config.params['indices']
	# Set up output directory and filename    
	# Create filename from focus landmarks
	if len(indices) == 0:
		indices = None
		focus_str = 'all'
	else:
		focus_str = ','.join([str(x) for x in list(indices)])

	out_plot = config.params['plot_dir'] + focus_str + '.png'

	print(f"Output plot will be saved to: {out_plot}")
	
	# Ensure output directory exists
	os.makedirs(config.params['plot_dir'], exist_ok=True)
	
	print(f"Focus landmarks (indices): {indices}")
	# Create the plot

	shared_kwargs = dict(
		focus_on=indices,
		element=config.params['scatter_element'],
		landmarkSize=config.params['landmarkSize'],
		transformation=config.params['transformation'],
		alpha_actual=config.params['alpha'],
		percent_size=config.params['percent_size'],
		dpi=config.params['dpi'],
		size_threshold=config.params['num_labels'],
		ratio_threshold=config.params['ratio_threshold'],
		remove_clutter=config.params['remove_clutter'],
		pad_pixels=config.params['pad_pixels'],
		out_file=out_plot,
	)

	if config.params['point_from_file']:
		plot_points_from_file(
			config.params['point_from_file'],
			config.params['plotted_points'],
			points,
			centroids,
			voltages,
			label_counts,
			centroid_others,
			**shared_kwargs,
		)
	else:
		plot_landmark_subset(
			points,
			centroids,
			label_counts,
			centroid_others=centroid_others,
			**shared_kwargs
		)

	
	print(f"Visualization completed and saved to: {out_plot}")
	
	# Print shapes for verification (equivalent to the last cell)
	print(f"Final verification - Centroids shape: {centroids.shape}, Points shape: {voltages.voltage_array().shape}")


if __name__ == "__main__":
	main()
