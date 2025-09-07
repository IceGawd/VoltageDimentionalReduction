import sys

#set path for imports to include ../clean_code
sys.path.append('../clean_code/')

from Utilities import config
from Utilities.set_params import set_params
from main import main as main_main
from select_landmarks_MI import main as select_main


set_params()

main_main()
select_main()

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