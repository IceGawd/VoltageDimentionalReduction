set -e

python ../clean_code/main.py \
	../../Voltage_Data/synthetic/swiss_roll.csv\
	--max_centroids 1000 \
	--batch_size=1000 \
	--split_char ',' \
	--save_data ../../Voltage_Temp/Results/synthetic/saved_data.pkl

python ../clean_code/select_landmarks_MI.py \
	--NoOfLandmarks 20 \
	--save_data ../../Voltage_Temp/Results/synthetic/saved_data.pkl

python ../clean_code/Visualization/visualizations.py \
	--scatter_element point \
	--save_data ../../Voltage_Temp/Results/mnist/saved_data.pkl \
	--plot_dir ../../Voltage_Temp/Scatter_Plots/synthetic/ \
	--dpi 300