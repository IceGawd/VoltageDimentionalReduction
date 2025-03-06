import kmeans
import create_data
import time
import voltage
import matplotlib.pyplot as plt

if __name__ == '__main__':
	iters = 200
	maximum = 1000000
	k = 1000

	inc = maximum // iters

	value = []
	times = []

	for points in range(inc, maximum, inc):
		value.append(points)

		create_data.create_dataset_eigth_sphere(output_file="../inputoutput/data/temp.json", points=points, seed=time.time(), stream=True)
		data = create_data.Data("../inputoutput/data/temp.json", stream=True)

		start = time.time()

		partitions = Partitions(data)
		partitions.k_means(k, seed=time.time())

		end = time.time()

		times.append(end - start)

	plt.figure(figsize=(10, 5))
	plt.plot(value, times, marker='o', linestyle='-', color='b', label="K-Means Runtime")

	plt.xlabel("Number of Points")
	plt.ylabel("Time Taken (seconds)")
	plt.title("K-Means Clustering Runtime vs. Dataset Size")
	plt.legend()
	plt.grid(True)

	plt.show()

	plt.savefig("../inputoutput/matplotfigures/kmeansVSsizeTimeGraph.png")