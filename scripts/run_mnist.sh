#set -e

# python ../clean_code/main.py \
# 	../../Voltage_Data/mnist/mnist.csv \
# 	--max_centroids 500 \
# 	--batch_size=10000 \
# 	--split_char ',' \
# 	--save_data ../../Voltage_Temp/Results/mnist/saved_data.pkl

# python ../clean_code/select_landmarks_MI.py \
# 	../../Voltage_Data/mnist/mnist.csv \
# 	--NoOfLandmarks 10 \
# 	--save_data ../../Voltage_Temp/Results/mnist/saved_data.pkl

python ../clean_code/Visualization/visualizations.py \
	../../Voltage_Data/mnist/mnist.csv \
	--scatter_element digit \
	--save_data ../../Voltage_Temp/Results/mnist/saved_data.pkl \
	--transformation umap \
	--plotted_points 2000 \
	--remove_clutter \
	--plot_dir ../../Voltage_Temp/Scatter_Plots/mnist/ \
	--plot_file mnist.png \
	--dpi 300
