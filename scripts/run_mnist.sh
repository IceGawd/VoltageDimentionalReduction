python ../clean_code/main.py \
       ../../Voltage_Data/mnist/mnist.csv\
       --split_char ',' \
       --save_data ../../Voltage_Temp/Results/mnist/saved_data.pkl

python ../clean_code/select_landmarks_MI.py \
       --save_data ../../Voltage_Temp/Results/mnist/saved_data.pkl \
       --NoOfLandmarks 20

