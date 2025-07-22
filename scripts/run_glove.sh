python ../clean_code/main.py \
../../Voltage_Data/glove/glove_with_pos.txt \
--split_char ' ' \
--normalize_vecs \
--save_data ../../Voltage_Temp/Results/glove/save_file.pkl

python ../clean_code/select_landmarks_MI.py \
--NoOfLandmarks 20 \
--save_data ../../Voltage_Temp/Results/glove/save_file.pkl

