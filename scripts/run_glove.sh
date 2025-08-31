python ../clean_code/main.py \
../../Voltage_Data/glove/short_glove_with_POS.csv \
--split_char ' ' \
--save_data ../../Voltage_Temp/Results/glove/save_file.pkl \
--batch-size 1000 \
--normalize_vec \
--max-centroids 1000
# --equalize_centroids

python ../clean_code/select_landmarks_MI.py \
       ../../Voltage_Data/glove/short_glove_with_POS.csv \
       --split_char ',' \
       --normalize_vec \
       --NoOfLandmarks 10 \
       --save_data ../../Voltage_Temp/Results/glove/save_file.pkl

