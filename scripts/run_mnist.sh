set -e

# python3 ../clean_code/main.py \
# 	../../Voltage_Data/mnist/mnist.csv \
# 	--save_data ../../Voltage_Temp/Results/mnist/saved_data.pkl \
# 	--split_char ',' \

# python3 ../clean_code/select_landmarks_MI.py \
# 	--save_data ../../Voltage_Temp/Results/mnist/saved_data.pkl \
# 	--NoOfLandmarks 10 \

python3 ../clean_code/Visualization/visualizations.py \
	--save_data ../../Voltage_Temp/Results/mnist/saved_data.pkl \
	--plot_dir ../../Voltage_Temp/Scatter_Plots/mnist/centroid \
 	--pad_pixels 0 \
 	--distinct_colors \
	--remove_clutter

# python3 ../clean_code/Visualization/visualizations.py \
# 	--save_data ../../Voltage_Temp/Results/mnist/saved_data.pkl \
# 	--plot_dir ../../Voltage_Temp/Scatter_Plots/mnist/point \
# 	--point_from_file ../../Voltage_Data/mnist/mnist.csv \
# 	--dpi 300 \
# 	--percent_size 0.005

# python3 ../clean_code/Visualization/visualizations.py \
# 	--save_data ../../Voltage_Temp/Results/mnist/saved_data.pkl \
# 	--plot_dir ../../Voltage_Temp/Scatter_Plots/mnist/pointclutterless \
# 	--point_from_file ../../Voltage_Data/mnist/mnist.csv \
# 	--dpi 300 \
# 	--percent_size 0.005 \
# 	--pad_pixels 0 \
# 	--remove_clutter