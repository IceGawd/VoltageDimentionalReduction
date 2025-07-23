python ../clean_code/main.py \
../../Voltage_Data/glove/glove_with_pos.txt \
--split_char ' ' \
--save_data ../../Voltage_Temp/Results/glove/save_file.pkl \
--batch-size 40000 \
--normalize_vec \
--max-centroids 500
#--equalize_centroids

#python ../clean_code/select_landmarks_MI.py \
#--save_data ../../Voltage_Temp/Results/glove/save_file.pkl

