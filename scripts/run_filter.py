"""
A script to filter a large dataset (a csv file) using a list of 
landmarks indicated by their indices. Outputs to a csv file whose name is the original
name concatenated with a comma separated list of indices.

Usage:
run_mnistPartial.py <input file path> <--save_data <data_file_path>> <--indices indices>

Example:

python run_filter.py ../../Voltage_Temp/Results/mnist/mnist.csv\
     --data --save_data ../../Voltage_Temp/Results/mnist/saved_data.pkl\
     --indices 5  
"""
import sys
import subprocess
from pathlib import Path
from run_command import run_command

def main():
    """
    Main function to parse command-line arguments and run filter.py with appropriate parameters.
    """
    import argparse

    print("Starting run_filter.py")
    parser = argparse.ArgumentParser(
        description="Run filter.py on a given input file with specified voltage data and landmark indices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python run_filter.py ../../Voltage_Temp/Results/mnist/mnist.csv \\
    --data_file_path ../../Voltage_Temp/Results/mnist/saved_data.pkl \\
    --indices 7 10 14 15 16

This will filter the input CSV file using landmarks at indices 7, 10, 14, 15, 16
and create an output file named mnist_7,10,14,15,16.csv in the same directory.
        """
    )   
    
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('--data_file_path', required=True, 
                       help='Path to the pickle file containing voltage data')
    parser.add_argument('--indices', nargs='+', required=False, type=int,
                       help='List of landmark indices to filter by')
    parser.add_argument('--no_filter', action='store_true',
                       help='If set, disables filtering by indices')

    args = parser.parse_args()
    
    input_file = args.input_file
    data_file_path = args.data_file_path
    if not args.no_filter:
        # Construct output file name
        indices = [str(idx) for idx in args.indices]  # Convert back to strings for compatibility
 
        indices_str = ','.join(indices)
        input_path_obj = Path(input_file)
        output_file = input_path_obj.parent / f"{input_path_obj.stem}_{indices_str}{input_path_obj.suffix}"

        # Construct the command to run filter.py
        command_filter = [
            sys.executable, '../clean_code/filter.py',
            input_file,
            '--output_path', str(output_file),
            '--save_data', data_file_path,
            '--indices'
        ] + indices

        success = run_command(command_filter, "run filter.py")
        if not success:
            sys.exit(1)

        data_path_obj = Path(data_file_path)
        sub_data_file_path = data_path_obj.parent / f"{data_path_obj.stem}_{indices_str}{data_path_obj.suffix}"

    else:
        sub_data_file_path = data_file_path
        output_file = input_file

    command_main = [
        sys.executable, '../clean_code/main.py',
        str(output_file),
        '--save_data', str(sub_data_file_path),
        '--init-size', '1000',
        '--batch-size', '1000',
        '--max-centroids', '1000',
        '--split_char', ',']

    description = "running kmeans and solver"

    success = run_command(command_main, description)
    if not success:
        sys.exit(1)


    command_select = [
        sys.executable, '../clean_code/select_landmarks_MI.py',
        str(output_file),    
        '--NoOfLandmarks', '10',
        '--save_data', str(sub_data_file_path),
    ]

    description = "running landmark selection"

    success = run_command(command_select, description)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
