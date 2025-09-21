set -e

DATASET="swiss_roll_noisy"

python ../clean_code/main.py \
	../../Voltage_Data/synthetic/${DATASET}.csv \
	--max_centroids 100 \
	--batch_size=1000 \
	--split_char ',' \
	--save_data ../../Voltage_Temp/Results/synthetic/${DATASET}_saved_data.pkl

python ../clean_code/select_landmarks_MI.py \
	../../Voltage_Data/synthetic/${DATASET}.csv \
	--NoOfLandmarks 10 \
	--save_data ../../Voltage_Temp/Results/synthetic/${DATASET}_saved_data.pkl

python ../clean_code/Visualization/visualizations.py \
	../../Voltage_Data/synthetic/${DATASET}.csv \
	--continous_label \
	--scatter_element point \
	--save_data ../../Voltage_Temp/Results/synthetic/${DATASET}_saved_data.pkl \
	--plot_dir ../../Voltage_Temp/Scatter_Plots/synthetic/ \
	--plot_file ${DATASET}.png \
	--plotted_points 10000 \
	--dpi 300
