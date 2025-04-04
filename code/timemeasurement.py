import kmeans
import create_data
import time
import voltage
import matplotlib.pyplot as plt

if __name__ == '__main__':
	iters = 10
	maximum = 750

	# k = 100
	# points = 10000

	inc = maximum // iters

	value = []
	approx_times = []
	compute_times = []

	for points in range(inc, maximum, inc):
		print(points)

		value.append(points)

		create_data.create_dataset_line(output_file="../inputoutput/data/temp.json", start=0, end=3, points=points, seed=time.time(), stream=True)

		# data = create_data.create_dataset_eigth_sphere(points=points, seed=time.time())
		# create_data.rotate_into_dimention(data, higher_dim=dim, seed=time.time()).save_data("../inputoutput/data/temp.json")

		data = create_data.Data("../inputoutput/data/temp.json", stream=False)

		X0 = []
		X1 = []

		for index, point in enumerate(data):
			if point[0] < 1:
				X0.append(voltage.Landmark(index, 0))
			if point[0] > 2:
				X1.append(voltage.Landmark(index, 1))


		ungrounded = voltage.Solver(data)
		ungrounded.setWeights(voltage.gaussiankernel, 0.03)
		ungrounded.addLandmarks(X0)
		ungrounded.addLandmarks(X1)
		ungrounded.compute_voltages()

		start = time.time()
		ungrounded_voltage = ungrounded.compute_voltages()
		end = time.time()

		compute_times.append(end - start)

		ungrounded = voltage.Solver(data)
		ungrounded.setWeights(voltage.gaussiankernel, 0.03)
		ungrounded.addLandmarks(X0)
		ungrounded.addLandmarks(X1)
		ungrounded.compute_voltages()

		start = time.time()
		ungrounded_voltage = ungrounded.approximate_voltages(max_iters=10)
		end = time.time()

		approx_times.append(end - start)

		"""
		grounded = voltage.Solver(data)
		grounded.setWeights(voltage.gaussiankernel, 0.03)
		grounded.addLandmarks(X1)
		grounded.addUniversalGround()
		# grounded_voltage = grounded.compute_voltages()

		start = time.time()
		grounded_voltage = grounded.compute_voltages()
		end = time.time()

		compute_times.append(end - start)

		grounded = voltage.Solver(data)
		grounded.setWeights(voltage.gaussiankernel, 0.03)
		grounded.addLandmarks(X1)
		grounded.addUniversalGround()
		# grounded_voltage = grounded.compute_voltages()

		start = time.time()
		grounded_voltage = grounded.approximate_voltages(max_iters=10)
		end = time.time()

		approx_times.append(end - start)
		"""

	plt.figure(figsize=(10, 5))
	plt.plot(value, compute_times, marker='o', linestyle='-', color='purple', label="Full Compute")
	plt.plot(value, approx_times, marker='*', linestyle='--', color='green', label="Approximation")

	plt.xlabel("Points")
	plt.ylabel("Time Taken (seconds)")
	plt.title("Ungrounded vs. Points")
	plt.legend()
	plt.grid(True)

	plt.show()

	plt.savefig("../inputoutput/matplotfigures/ungroundedvoltageapproxVSpointsTimeGraph.png")