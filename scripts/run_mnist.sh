set -e

# python ../clean_code/main.py \
# 	../../Voltage_Data/mnist/mnist.csv\
# 	--max_centroids 2000 \
# 	--batch_size=10000 \
# 	--split_char ',' \
# 	--save_data ../../Voltage_Temp/Results/mnist/saved_data.pkl

# python ../clean_code/select_landmarks_MI.py \
# 	--NoOfLandmarks 20 \
# 	--save_data ../../Voltage_Temp/Results/mnist/saved_data.pkl

python ../clean_code/Visualization/visualizations.py \
	--point_from_file ../../Voltage_Data/mnist/mnist.csv \
	--plotted_points 1000 \
	--scatter_element digit \
	--save_data ../../Voltage_Temp/Results/mnist/saved_data.pkl \
	--plot_dir ../../Voltage_Temp/Scatter_Plots/mnist/ \
	--dpi 300