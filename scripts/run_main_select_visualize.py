import os
import sys

sys.path.append(os.path.abspath('../clean_code/'))

from Utilities.set_params import set_params
import Utilities.config as config
from main import main as main_main
from select_landmarks_MI import main as select_main

print("imported")
sys.exit()
def execute(fullname):
    print(f"Executing run_main_select.executewith fullname: {fullname}")
    if fullname == '':
        print("Error: fullname is empty")
        sys.exit(1)
        
    # if fullname contains '_', split it on first '_' into name and suffix
    if '_' in fullname:
        name, suffix = fullname.split('_', 1)
        suffix='_'+suffix
    else:
        name = fullname
        suffix = ''

    print(f"name: {name}, suffix: {suffix}")

    set_params()
    
    config.params['filter_partition'] = True
    config.params['normalize_vec'] = name=='glove'
    config.params['file_path'] = \
        f'../../Voltage_Data/{name}/{name}{suffix}.csv'
    config.params['save_data'] = \
        f'../../Voltage_Temp/Results/{name}/saved_data{suffix}.pkl'
    config.params['plot_dir'] = \
        f'../../Voltage_Temp/Scatter_Plots/{name}{suffix}'
    config.params['output_path'] = \
        f'../../Voltage_Data/{name}/{name}{suffix}.csv'
    main_main()
    select_main()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_main_select_visualize.py <fullname>")
        sys.exit(1)
    
    fullname = sys.argv[1]
    execute(fullname)  

# command_visualize = [
#     sys.executable, 'visualizations.py',
#     '--data_file', str(data_file),
#     '--plot_dir', str(args.plot_dir)
# ]

# description = "running visualization"

# success = run_command(command_visualize, description)
# if not success:
#     sys.exit(1)

#