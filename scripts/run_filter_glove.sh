python ../clean_code/filter.py \
../../Voltage_Data/glove/glove_with_pos_label.csv \
       --split_char ' ' \
       --normalize_vec \
       --save_data ../../Voltage_Temp/Results/glove/save_file.pkl \
       --output_path ../../Voltage_Data/glove/glove.csv\
       --filter_partition


