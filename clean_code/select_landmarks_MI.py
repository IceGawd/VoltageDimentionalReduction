import voltagemap   
import numpy as np
from Utilities import config
from copy import copy
from Mutual_Information import mutual_information
from Utilities.set_params import set_params

def select_landmarks(all_voltages):
    """Selects landmarks as a subset of all_voltages. Uses gready search on the mutual information between 
    the landmarks and the identity of the centroid"""
    # Initialize the map
    voltage_map=copy(all_voltages)
    voltage_map.set_advantages(-np.inf,quantity="MI_cumul")  # Set initial advantages to negative infinity
    L=len(voltage_map.entries)
    N=30; #config.params['NoOfLandmarks']
    if N>L:
        raise ValueError(f"Number of landmarks {N} exceeds the number of available landmarks {L}.") 
    
    # initialize the quantity "MI_1" to be the mutual information for each landmark by itself
    for i in range(L):
        voltage_matrix = np.stack([voltage_map.entries[i]['voltages']],axis=1)
        # Compute mutual information (MI) between voltage_matrix and the identity of the centroid
        mi = mutual_information(voltage_matrix)
        voltage_map.entries[i]['MI_1'] = mi  # Set the advantage to the mutual
        if(i%10==0):
            print(f"Landmark index={i} MI_1={mi:.4f}",end='\r')

    voltage_map.sort_by_advantage(quantity="MI_1", reverse=True)  # Sort the voltage map by MI_1

    L= 100
    # repeatedly iteration all_voltages.entries and add the landmark with the largest distance to the selected landmarks to the voltage map	
    for i in range(N):
        # find the landmark whose addition to the voltage map will increase the mutual information by the max amount
        max_mi = -np.inf
        for j in range(i,L):
            voltage_matrix = np.stack([voltage_map.entries[k]['voltages'] for k in (list(range(i))+[j]) ],axis=1)  # Stack the voltages of the current landmarks and the new candidate landmark
            # Compute mutual information (MI) between voltage_matrix and the identity of the centroid
            mi = mutual_information(voltage_matrix)
            #print(f"Landmarks 0:{i}+{j} MI={mi:.4f}")
            if mi > max_mi:
                max_mi = mi
                best_landmark = j

        # move the best landmark to the voltage map
        print(f"Adding landmark index={best_landmark} lm_index={voltage_map.entries[best_landmark]['landmark'].index} with MI={max_mi:.4f}")
        voltage_map.entries[best_landmark]['MI_cumul'] = max_mi  # Update the advantage of the best landmark
        # Sort the voltage map by advantage
        voltage_map.sort_by_advantage(quantity="MI_cumul", reverse=True)

    # remove landmarks beyond N
    voltage_map.entries = voltage_map.entries[:N]   
    return voltage_map

if __name__ == "__main__": 
    set_params()  # Set parameters from command line or default values
    config.params['Voltage_map_output']= '../../Voltage_Temp/Results/voltage_map.npy'


    import pickle
    with open(config.params['Voltage_map_output'], 'rb') as f:
	    Data=pickle.load(f)
    # Load the data from the specified file     
    print(f"Data loaded from {config.params['Voltage_map_output']}")
    all_voltages = Data['all_voltages']
    voltage_map = select_landmarks(all_voltages)

    import pickle
    Data['voltage_map'] = voltage_map
    with open(config.params['Voltage_map_output'], 'wb') as f:
        pickle.dump(Data, f)
    print(f"added voltage_map to {config.params['Voltage_map_output']}")


