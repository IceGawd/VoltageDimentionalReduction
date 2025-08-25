python ../clean_code/main.py \
       ../../Voltage_Data/mnist/mnist7,10,14,15,16.csv\
       --init-size 1000\
       --batch-size 2000\
       --max-centroids 1000\
       --split_char ',' \
       --save_data ../../Voltage_Temp/Results/mnist/saved_data7,10,14,15,16.pkl

python ../clean_code/select_landmarks_MI.py \
       --NoOfLandmarks 10 \
       --save_data ../../Voltage_Temp/Results/mnist/saved_data7,10,14,15,16.pkl

