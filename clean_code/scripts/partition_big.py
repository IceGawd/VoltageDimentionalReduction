import os
import sys

# gets base as input and scans all files, processes all files that were not processed yet
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, '..')
sys.path.append(relative_path)

from Utilities.set_params import set_params
import Utilities.config as config
set_params()

from filter import main as filter_main

# find the file name and the number of lines in each file of the form path/name_*.ext
#find the number of lines in each file
import glob

base="glove"
pattern = f"../../Voltage_Data/{base}/{base}"
min_lines=10000

for f in glob.glob(pattern+"*.csv"):
    #split f into path, name and extension
    path, name = os.path.split(f)
    ext = os.path.splitext(f)[1]
    #remove extension from name
    name = name[:-len(ext)] if name.endswith(ext) else name

    # count number of lines in file f
    with open(f, 'r') as file:
        num_lines = sum(1 for line in file)
    print(f"File: {f}, Lines: {num_lines}")
    if "processed" in name and\
          not "partitioned" in name\
          and num_lines>min_lines:
        
        # remove _processed from name
        name = name.replace('.processed','')
        # partition name on the first '_' into name and suffix
        if '_' in name:
            name, suffix = name.split('_', 1)
            suffix='_'+suffix
        else:
            suffix = ''
        print(f"partitioning file: {name}")
        config.params['filter_partition'] = True
        config.params['file_path'] = f'../../Voltage_Data/{base}/{name}{suffix}.processed.csv'
        config.params['save_data'] = f'../../Voltage_Temp/Results/{base}/saved_data{suffix}.pkl'
        config.params['output_path'] = f'../../Voltage_Data/{base}/{name}{suffix}.csv'

        for key in ['file_path','save_data','output_path']:
            print(f"{key}: {config.params[key]}")

        filter_main()
    
        #after exec, we change the filename to indicate it was processed
        new_name = f.replace(ext, '.partitioned'+ext)
        os.rename(f, new_name)
        print(f"Renamed {f} to {new_name}")

    
