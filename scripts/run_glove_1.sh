python run_main_select_visualize.py \
        ../../Voltage_Data/glove/glove_with_pos_label.csv\
        --split_char ',' \
        --batch_size 10000 \
        --normalize_vecs \
        --save_data ../../Voltage_Temp/Results/glove/saved_data.pkl \
        --NoOfLandmarks 20\

# python ../clean_code/Visualization/visualizations.py --save_data ../../Voltage_Temp/Results/mnist/saved_data.pkl --plot_dir ../../Voltage_Temp/Scatter_Plots/mnist/
