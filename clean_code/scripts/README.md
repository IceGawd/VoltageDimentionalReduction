## run_main_select_visualize.py
given a single parameter that is mnist or glove (or another dataset)
followed by "_i_j_..." it runs the three steps on the file using the stem to define the relevant directories.

## run_filter_glove.sh

bash"""
python ../clean_code/filter.py \
../../Voltage_Data/glove/glove.csv \
       --split_char ' ' \
       --normalize_vec \
       --save_data ../../Voltage_Temp/Results/glove/save_file.pkl \
       --output_path ../../Voltage_Data/glove/glove.csv\
       --filter_partition
"""

## Run_all

runs msv on all versions of a given file.