""" A wrapper around mnistVisuals.py"""


import sys, os
sys.path.append(os.path.abspath("../clean_code/"))
from Utilities import config
from Utilities import set_params
set_params.set_params()

import pickle
import numpy as np

import visualHelpers
from mnistVisuals import plot_landmark_subset


def main():
    set_params.set_params()

    # Load data from the pickle file
    save_data=config.params['save_data']
    print(f"Loading data from {save_data}")
    
    with open(save_data, 'rb') as f:
        data = pickle.load(f)
    
    # Extract data components
    voltages = data['voltage_map']
    centroids = data['centroids']
    label_counts = data['label_counts']  
    
    indices = config.params['indices']
     # Set up output directory and filename    
    # Create filename from focus landmarks
    if len(indices)==0:
        focus_str = 'all'
    else:
        focus_str = ','.join([str(x) for x in list(indices)])

    out_plot = config.params['plot_dir'] + focus_str + '.png'

    print(f"Output plot will be saved to: {out_plot}")
    
    # Ensure output directory exists
    os.makedirs(config.params['plot_dir'], exist_ok=True)
    
    indices = config.params['indices']
    print(f"indices={indices}")
    print(f"Focus landmarks (indices): {indices}")
    # Create the plot
    plot_landmark_subset(
        voltages.voltage_array(), 
        centroids, 
        label_counts, 
        focus_on=indices,
        log_transform=not config.params['no_log_transform'],
        transformation='pca',
        out_file=out_plot,
    )
    
    print(f"Visualization completed and saved to: {out_plot}")
    
    # Print shapes for verification (equivalent to the last cell)
    print(f"Final verification - Centroids shape: {centroids.shape}, Points shape: {voltages.voltage_array().shape}")


if __name__ == "__main__":
    main()
