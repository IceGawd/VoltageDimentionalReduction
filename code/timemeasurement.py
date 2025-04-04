import kmeans
import create_data
import time
import voltage
import matplotlib.pyplot as plt

if __name__ == '__main__':
	iters = 50
	maximum = 2000

	# k = 100
	# points = 10000

	inc = maximum // iters

	value = []
	times = []

	for points in range(inc, maximum, inc):
		print(points)

		value.append(points)

		create_data.create_dataset_line(output_file="../inputoutput/data/temp.json", points=points, seed=time.time(), stream=True)

		# data = create_data.create_dataset_eigth_sphere(points=points, seed=time.time())
		# create_data.rotate_into_dimention(data, higher_dim=dim, seed=time.time()).save_data("../inputoutput/data/temp.json")

		data = create_data.Data("../inputoutput/data/temp.json", stream=True)

		start = time.time()

		# ungrounded = voltage.Solver(data)

		X0 = []
		X1 = []

		for index, point in enumerate(data):
			if point[0] < 1:
				X0.append(voltage.Landmark(index, 0))
			if point[0] > 2:
				X1.append(voltage.Landmark(index, 1))

		# ungrounded.setWeights(voltage.gaussiankernel, 0.03)
		# ungrounded.addLandmarks(X0)
		# ungrounded.addLandmarks(X1)
		# ungrounded.compute_voltages()

		grounded = voltage.Solver(data)
		grounded.setWeights(voltage.gaussiankernel, 0.3)
		grounded.addUniversalGround()
		grounded.addLandmarks(X1)
		# grounded_voltage = grounded.compute_voltages()
		grounded_voltage = approximate_voltages(max_iters=100)

		end = time.time()

		times.append(end - start)

	plt.figure(figsize=(10, 5))
	plt.plot(value, times, marker='o', linestyle='-', color='b', label="Runtime")

	plt.xlabel("Points")
	plt.ylabel("Time Taken (seconds)")
	plt.title("approximate_voltages(max_iters=100)Grounded vs. Points")
	plt.legend()
	plt.grid(True)

	plt.show()

	plt.savefig("../inputoutput/matplotfigures/groundedvoltageapproxVSpointsTimeGraph.png")