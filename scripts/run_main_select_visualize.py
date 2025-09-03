import sys
import subprocess
from pathlib import Path
from run_command import run_command

def main():
    """
    Main function to parse command-line arguments and run filter.py with appropriate parameters.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="run main+select+visualize.",
        formatter_class=argparse.RawDescriptionHelpFormatter)     
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument(
        '--data_file',
        default='../../Voltage_Temp/Results/mnist/error.pkl',
        help='Path to the pickle file containing voltage data'
    )
    parser.add_argument(
        '--plot_dir',
        default='../../Voltage_Temp/Scatter_Plots/mnist/',
        help='Directory to save output plots'
    )
    
    args = parser.parse_args()
    
    input_file = args.input_file
    data_file = args.data_file

    if True:
        command_main = [
            sys.executable, '../clean_code/main.py',
            str(input_file),
            '--save_data', str(data_file),
            '--init-size', '500',
            '--batch-size', '251',
            '--max-centroids', '1000',
            '--split_char', ',']

        description = "running kmeans and solver"

        success = run_command(command_main, description)
        if not success:
            sys.exit(1)


        command_select = [
            sys.executable, '../clean_code/select_landmarks_MI.py',
            str(input_file),    
            '--NoOfLandmarks', '10',
            '--save_data', str(data_file),
        ]

        description = "running landmark selection"

        success = run_command(command_select, description)
        if not success:
            sys.exit(1)

    command_visualize = [
        sys.executable, 'visualizations.py',
        '--data_file', str(data_file),
        '--plot_dir', str(args.plot_dir)
    ]

    description = "running visualization"

    success = run_command(command_visualize, description)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
