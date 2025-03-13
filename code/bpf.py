import voltage
import numpy as np

dist = []


def nInfUniform(voltages):
	voltages.sort()
	uniform = np.array([x / (len(voltages) - 1) for x in range(len(voltages))])

	return np.linalg.norm(abs(voltages - uniform))

def nInfExp(voltages, base=10):
	global dist

	voltages.sort()

	if (len(dist) != len(voltages)):
		dist = np.array([np.pow(base, (x / (len(voltages) - 1)) - 1) for x in range(len(voltages))])

	return np.linalg.norm(abs(voltages - dist))

def median(voltages, value=0.5):
	voltages.sort()
	return abs(voltages[int(len(voltages) / 2)] - value)

def minimum(voltages, value=0.1):
	voltages.sort()
	return abs(voltages[0] - value)

def minWithStd(voltages, value=0.1):
	voltages.sort()
	return abs(voltages[0] - value) / np.std(voltages)


def bestParameterFinder(kernel, landmarks, partition, metric=nInfUniform, emin=-5, emax=5, mantissa=True, divisions=1):
	"""
	Finds the best parameters (C and P_G) for a solver based on voltage distribution minimization.

	This function searches for optimal parameters `C` and `P_G` by iterating over exponent values in 
	a specified range, computing voltages using a solver, and minimizing some metric
	between the voltage distribution and a uniform distribution.

	Parameters:
	-----------
	kernel : object
		The kernel function or object used to compute partition weights.
	landmarks : list
		A list of landmark points used in the solver.
	partition : object
		A partition object containing centers used in the solver.
	nInfUniform : function (list of floating point values -> floating point value)
		A function that is used to quantify if voltages are good or bad, the smaller the better
	emin : int, optional (default=-5)
		The minimum exponent value to consider for `C` and `G` (10^emin).
	emax : int, optional (default=5)
		The maximum exponent value to consider for `C` and `G` (10^emax).
	mantissa : bool, optional (default=True)
		If True, performs an additional fine-tuning step by adjusting the mantissa of `C` and `G`.
	divisions : int, optional (default=1)
		The number of divisions used for refining `C` and `G` in the mantissa tuning phase.
	L : float, optional (default=0.3)
		A weight factor applied to the minimum voltage value in the optimization process.

	Returns:
	--------
	tuple
		A tuple (bestC, bestG), where:
		- bestC (float): The optimized value for parameter C.
		- bestG (float): The optimized value for parameter P_g.
	"""

	bestC = emin
	bestG = emin

	val = float('inf')

	for c_e in range(emin, emax+1):
		for g_e in range(emin, emax+1):
			# print(e)
			meanSolver = voltage.Solver(partition.centers)
			meanSolver.setPartitionWeights(kernel, partition, pow(10, c_e))
			meanSolver.addUniversalGround(pow(10, g_e))
			meanSolver.addLandmarks(landmarks)

			voltages = np.array(meanSolver.compute_voltages())
			tempval = metric(voltages)

			# print(tempval)
			if (val > tempval):
				bestC = c_e
				bestG = g_e
				val = tempval

	bestc = -9
	bestg = -9
	val = float('inf')

	if (mantissa):
		for c in range(-9, 10, divisions):
			C = pow(10, bestC) + c * pow(10, bestC - 1)

			for g in range(-9, 10, divisions):
				G = pow(10, bestG) + g * pow(10, bestG - 1)

				# print(v)
				meanSolver = voltage.Solver(partition.centers)
				meanSolver.setPartitionWeights(kernel, partition, C)		
				meanSolver.addUniversalGround(G)
				meanSolver.addLandmarks(landmarks)

				voltages = np.array(meanSolver.compute_voltages())
				tempval = metric(voltages)

				if (val > tempval):
					bestc = c
					bestg = g
					val = tempval
	else:
		bestc = 0
		bestg = 0

	return pow(10, bestC) + bestc * pow(10, bestC - 1), pow(10, bestG) + bestg * pow(10, bestG - 1)

if __name__ == "__main__":
	 for metric in [nInfUniform, nInfUniform, median, minimum, minWithStd]:
	 	for kernel in [voltage.gaussiankernel, voltage.radialkernel]:
			c, p_g = bpf.bestParameterFinder(voltage.gaussiankernel, landmarks, partitions, metric=bpf.nInfUniform, emin=-6, emax=-1, mantissa=False, divisions=3)