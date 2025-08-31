#!/usr/bin/env python3
"""
Voltage Visualization Script (from notebook)

This script loads voltage data from a pickle file and creates visualizations
of voltage maps with landmark subsets. Converted from yoav_Visualizations.ipynb notebook.

Usage:
    python visualizations_notebook.py
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Add the clean_code directory to the Python path
sys.path.append(os.path.abspath("../clean_code/"))

# Change to clean_code directory (equivalent to %cd ../clean_code/)
os.chdir("../clean_code/")

# Import required modules
from Visualization import visualHelpers
from Visualization.mnistVisuals import plot_landmark_subset


def main():
    """
    Main function that replicates the notebook functionality with command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create voltage visualizations from MNIST data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python visualizations.py \\
    --data_file ../../Voltage_Temp/Results/mnist/saved_data_7.pkl \\
    --plot_dir ../../Voltage_Temp/Scatter_Plots/mnist/ \\
    --indices 1 9 18

  python visualizations.py \\
    --data_file ../../Voltage_Temp/Results/mnist/saved_data.pkl \\
    --indices 7 10 14 15 16
        """
    )
    
    parser.add_argument(
        '--data_file',
        default='../../Voltage_Temp/Results/mnist/saved_data_7.pkl',
        help='Path to the pickle file containing voltage data'
    )
    
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='If set, display plots interactively'
    )

    parser.add_argument(
        '--plot_dir',
        default='../../Voltage_Temp/Scatter_Plots/mnist/',
        help='Directory to save output plots'
    )
    
    parser.add_argument(
        '--indices',
        nargs='+',
        type=int,
        default=[],
        help='List of landmark indices to focus on (focus_on in notebook)'
    )
    
    args = parser.parse_args()
    
    # Extract arguments
    data_file = args.data_file
    plots_dir = args.plot_dir
    focus_on = np.array(args.indices)
    
    print("Starting voltage visualization")
    print(f"Data file: {data_file}")
    print(f"Plot directory: {plots_dir}")
    print(f"Focus landmarks (indices): {focus_on}")
    
    # Load data from the pickle file
    print(f"Loading data from {data_file}")
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract data components
    voltages = data['voltage_map']
    centroids = data['centroids']
    label_counts = data['label_counts']
    
    print(f"Data loaded successfully")
    print(f"Available data keys: {list(data.keys())}")
    
    print(f"Focus landmarks: {focus_on}")
    
    # Get voltage points
    points = voltages.voltage_array()
    
    # Transform points using PCA
    transformed_points = visualHelpers.transform(points, "pca")
    
     
    # Set up output directory and filename (now from command line)
    # plots_dir is now set from arguments
    
    # Create filename from focus landmarks
    if len(focus_on)==0:
        focus_str = 'all'
    else:
        focus_str = ','.join([str(x) for x in list(focus_on)])

    out_plot = plots_dir + focus_str + '.png'

    print(f"Output plot will be saved to: {out_plot}")
    
    # Ensure output directory exists
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set visualization parameters
    landmarkSize = 3
    alpha_actual = 1
    percent_size = 0.02
    log_transform = True
    num_labels = 10
    
    print("Creating visualization...")
    
    # Create the plot
    plot_landmark_subset(
        points, 
        centroids, 
        label_counts, 
        focus_on, 
        percent_size=percent_size,
        alpha_actual=alpha_actual, 
        out_file=out_plot,
        element='digit'
    )
    
    print(f"Visualization completed and saved to: {out_plot}")
    
    # Print shapes for verification (equivalent to the last cell)
    print(f"Final verification - Centroids shape: {centroids.shape}, Points shape: {points.shape}")


if __name__ == "__main__":
    main()
