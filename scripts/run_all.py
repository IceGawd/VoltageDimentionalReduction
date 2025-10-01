
dataset="synthetic"
post="torus"
no_of_landmarks=4
main_command="""python ../clean_code/main.py \
	../../Voltage_Data/{dataset}/{post}.csv\
	--max_centroids 100 \
	--init_size 100 \
	--batch_size 500 \
	--split_char ',' \
	--save_data ../../Voltage_Temp/Results/{dataset}/saved_data{post}.pkl
"""

select_command = """python ../clean_code/select_landmarks_MI.py \
	../../Voltage_Data/{dataset}/{post}.csv\
	--NoOfLandmarks {no_of_landmarks}\
	--save_data ../../Voltage_Temp/Results/{dataset}/saved_data{post}.pkl
"""

visualize_command="""python ../clean_code/Visualization/visualizations.py \
	../../Voltage_Data/{dataset}/{post}.csv \
	--scatter_element point \
	--continous_label \
	--save_data ../../Voltage_Temp/Results/{dataset}/saved_data{post}.pkl \
	--plotted_points 5000 \
	--percent_size 0.012 \
	--alpha 0.8 \
	--plot_file ../../Voltage_Temp/Scatter_Plots/{dataset}/{dataset}{post}.png \
	--plot_dir ../../Voltage_Temp/Scatter_Plots/{dataset}/ \
	--dpi 300
	"""

filter_command="""python ../clean_code/filter.py \
	   ../../Voltage_Data/{dataset}/{post}.csv \
	   --batch_size=400 \
	   --output_path ../../Voltage_Data/{dataset}/{post}.csv\
	   --save_data ../../Voltage_Temp/Results/{dataset}/saved_data{post}.pkl \
	   --filter_partition
	   """


import os
def run_command(command, description):
	print(f"Starting: {description}")
	result = os.system(command)
	if result != 0:
		print(f"Error during: {description}")
		return False
	print(f"Completed: {description}")
	return True

def check_file_size(file_path, min_line_no=100):
	""" Check if a file exists and has at least min_line_no lines"""
	if not os.path.isfile(file_path):
		return False
	with open(file_path, 'r') as f:
		lines = f.readlines()
	print(f"File {file_path} has {len(lines)} lines")
	return len(lines) >= min_line_no


# run_command(main_command.format(post=post,dataset=dataset), "Running main command")
# run_command(select_command.format(post=post,dataset=dataset, no_of_landmarks=no_of_landmarks), "Running select command")
# run_command(visualize_command.format(post=post,dataset=dataset), "Running visualize command")
# run_command(filter_command.format(post=post,dataset=dataset), "Running filter command")

for i in range(no_of_landmarks):
	filter_post = post + "_" + str(i)
	print("="*50, filter_post)
	if (check_file_size(f"../../Voltage_Data/{dataset}/{filter_post}.csv", 0)):
		run_command(main_command.format(post=filter_post,dataset=dataset), "Running main command")
		run_command(select_command.format(post=filter_post,dataset=dataset, no_of_landmarks=4), "Running select command")
		run_command(visualize_command.format(post=filter_post,dataset=dataset), "Running visualize command")
		#run_command(filter_command.format(post=post,dataset=dataset), "Running filter command")
	
import sys
sys.exit(0)

for i in range(10):
	for j in range(10):
		post=f"_{i}_{j}"
		print("="*50, post)

		if(check_file_size(f"../../Voltage_Data/{dataset}/{post}.csv")):

			run_command(main_command.format(post=post,dataset=dataset), "Running main command")
			run_command(select_command.format(post=post,dataset=dataset), "Running select command")
			run_command(visualize_command.format(post=post,dataset=dataset), "Running visualize command")
			run_command(filter_command.format(post=post,dataset=dataset), "Running filter command")

for i in range(10):
	for j in range(10):
		for k in range(10):
			post=f"_{i}_{j}_{k}"
			print("="*50, post)

			if(check_file_size(f"../../Voltage_Data/{dataset}/{post}.csv")):

				run_command(main_command.format(post=post,dataset=dataset), "Running main command")
				run_command(select_command.format(post=post,dataset=dataset), "Running select command")
				run_command(visualize_command.format(post=post,dataset=dataset), "Running visualize command")
 