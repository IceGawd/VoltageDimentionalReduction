import os
import sys

# gets base as input and scans all files, processes all files that were not processed yet
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, '..')
sys.path.append(relative_path)

from Utilities.set_params import set_params
import Utilities.config as config
set_params()

from scripts.run_main_select_visualize import execute

# find the file name and the number of lines in each file of the form path/name_*.ext
#find the number of lines in each file
import glob

base='glove'
pattern = f"../../Voltage_Data/{base}/{base}"


while True:
    found_file = False
    print("Scanning for files to process...")
    #find a file to process
    for f in glob.glob(pattern+"*.csv"):
        #split f into path, name and extension
        path, name = os.path.split(f)
        ext = os.path.splitext(f)[1]
        #remove extension from name
        name = name[:-len(ext)] if name.endswith(ext) else name
        with open(f, 'r') as file:
            num_lines = sum(1 for line in file) 
        # check if file name f"_{f}" exists, if it does, skip processing
        flag_file_name=f"{path}/_{name}{ext}"

        if os.path.exists(flag_file_name): 
            print(f"Skipping because {flag_file_name} exists")
            continue
        # if name contains 'processed', or file too small skip processing
        if "processed" in name or num_lines<10000:
            continue
        found_file = True
        break
    if found_file:
        print(f"Processing file: {name} num_lines: {num_lines}")
        # create an empty flag file
        print(f"Creating flag file: {flag_file_name}")
        with open(flag_file_name,'w') as flag_file:
            flag_file.write('')
        execute(name)
        #after exec, we change the filename to indicate it was processed
        new_name = f.replace(ext, '.processed'+ext)
        os.rename(f, new_name)
        print(f"Renamed {f} to {new_name}")
        #delete the flag file
        os.remove(flag_file_name)
    else:
        #wait 10 seconds and try again
        print("No more files to process, waiting 10 seconds")
        import time
        time.sleep(10)
    
