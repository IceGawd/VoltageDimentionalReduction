set -e

python ../clean_code/main.py \
	../../Voltage_Data/glove/glove.csv \
	--save_data ../../Voltage_Temp/Results/glove/saved_data.pkl \
	--batch_size 1000 \
	--normalize_vec \
	--max_centroids 2000
	# --equalize_centroids

python ../clean_code/select_landmarks_MI.py \
	../../Voltage_Data/glove/glove.csv \
	--save_data ../../Voltage_Temp/Results/glove/saved_data.pkl \
	--split_char ',' \
	--normalize_vec \
	--NoOfLandmarks 20 \

python ../clean_code/Visualization/visualizations.py \
	--save_data ../../Voltage_Temp/Results/glove/saved_data.pkl \
	--plot_dir ../../Voltage_Temp/Scatter_Plots/glove/centroid \
	--scatter_element point \
	--ratio_threshold 0.5

python ../clean_code/Visualization/visualizations.py \
	--save_data ../../Voltage_Temp/Results/glove/saved_data.pkl \
	--plot_dir ../../Voltage_Temp/Scatter_Plots/glove/points \
	--point_from_file ../../Voltage_Data/glove/glove.csv \
	--scatter_element word \
	--dpi 300 \
	--percent_size=0.005 \
	--remove_clutter

