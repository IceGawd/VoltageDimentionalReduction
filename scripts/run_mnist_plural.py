

import os
for i in range(10):
    post=f"_{i}"
    print("="*50, post)
    command=f"""python ../clean_code/main.py \
        ../../Voltage_Data/mnist/{post}/mnist.csv\
        --max_centroids 100 \
        --batch_size=1000 \
        --split_char ',' \
        --save_data ../../Voltage_Temp/Results/mnist/saved_data{post}.pkl
    """

    print(command)
    os.system(command)

    command = f"""python ../clean_code/select_landmarks_MI.py \
        --NoOfLandmarks 10 \
        --save_data ../../Voltage_Temp/Results/mnist/saved_data{post}.pkl
    """
    print(command)
    os.system(command)

    command=f"""python ../clean_code/Visualization/visualizations.py \
        --scatter_element digit \
        --save_data ../../Voltage_Temp/Results/mnist/saved_data{post}.pkl \
        --point_from_file ../../Voltage_Data/mnist/{post}/mnist.csv \
        --plotted_points 2000 \
        --plot_dir ../../Voltage_Temp/Scatter_Plots/mnist{post}/ \
        --dpi 300
        """

    print(command)
    os.system(command)
