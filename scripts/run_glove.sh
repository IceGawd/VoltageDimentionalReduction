set -e

# python3 ../clean_code/main.py \
# 	../../Voltage_Data/glove/short_glove_with_POS.csv \
# 	--save_data ../../Voltage_Temp/Results/glove/saved_data.pkl \
# 	--batch_size 1000 \
# 	--normalize_vec \
# 	--max_centroids 1000
# 	# --equalize_centroids

# python3 ../clean_code/select_landmarks_MI.py \
# 	../../Voltage_Data/glove/short_glove_with_POS.csv \
# 	--save_data ../../Voltage_Temp/Results/glove/saved_data.pkl \
# 	--split_char ',' \
# 	--normalize_vec \
# 	--NoOfLandmarks 10 \

python3 ../clean_code/Visualization/visualizations.py \
	--save_data ../../Voltage_Temp/Results/glove/saved_data.pkl \
	--plot_dir ../../Voltage_Temp/Scatter_Plots/glove/centroid \
	--scatter_element word \
	--remove_clutter

python3 ../clean_code/Visualization/visualizations.py \
	--save_data ../../Voltage_Temp/Results/glove/saved_data.pkl \
	--plot_dir ../../Voltage_Temp/Scatter_Plots/glove/points \
	--point_from_file ../../Voltage_Data/glove/short_glove_with_POS.csv \
	--scatter_element word \
	--dpi 300 \
	--percent_size=0.05 \
	--remove_clutter