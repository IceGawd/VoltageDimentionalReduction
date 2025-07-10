import voltagemap   
import numpy as np

def select_landmarks(all_voltages):
    """Selects landmarks as a subset of all_voltages.
    This is a highly experimental function, all are invited to improve on it"""
    # Initialize the map
    voltage_map=voltagemap.VoltageMap()
    lm, voltages, _ = all_voltages.entries[0]  # get the first landmark and its voltages
    voltage_map.add_solution(lm, voltages=voltages)
    max_voltage=np.zeros(len(all_voltages))  # to keep track of the maximum voltage for each landmark

    # repeatedly iteration all_voltages.entries and add the landmark with the largest distance to the selected landmarks to the voltage map	
    for iteration in range(100):
        # Find the landmark in all_voltages.entries that is farthest from the current voltage_map entries
        max_min_dist = 2.0
        best_idx = None
        best_norm = 2.0
        for idx, (lm, voltages, norm) in enumerate(all_voltages.entries):
            # Skip if already in voltage_map
            if any(np.array_equal(lm.index, vmap_lm.index) for vmap_lm, _, _ in voltage_map.entries):
                continue	
            # Compute minimum distance to any entry in voltage_map
            min_dist = np.min([np.linalg.norm(voltages - vm[1]) for vm in voltage_map.entries])
            if min_dist > max_min_dist and norm> best_norm:
                max_min_dist = min_dist
                best_idx = idx
                best_norm = norm
        print(f"Iteration {iteration}: Best landmark index {best_idx} norm={best_norm:.4f} with min distance {max_min_dist:.4f}")
        if best_idx is not None:
            lm, voltages, norm = all_voltages.entries[best_idx]
            voltage_map.add_solution(lm, voltages=voltages)
        else:
            break
