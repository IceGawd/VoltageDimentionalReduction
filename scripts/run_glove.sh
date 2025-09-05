python ../clean_code/main.py \
       ../../Voltage_Data/glove/glove_with_pos_label.csv \
       --save_data ../../Voltage_Temp/Results/glove/saved_data.pkl \
       --batch_size 10000 \
       --normalize_vec \
       --max_centroids 1000
# --equalize_centroids

python ../clean_code/select_landmarks_MI.py \
       ../../Voltage_Data/glove/glove_with_pos_label.csv \
       --split_char ',' \
       --normalize_vec \
       --NoOfLandmarks 10 \
       --save_data ../../Voltage_Temp/Results/glove/saved_data.pkl

# python visualizations.py --save_data ../../Voltage_Temp/Results/glove/saved_data.pkl --plot_dir ../../Voltage_Temp/Scatter_Plots/glove/ --scatter_element point
