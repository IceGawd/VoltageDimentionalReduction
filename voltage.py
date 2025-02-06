from create_data import *
import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import solve

def compute_voltages(W, X0, X1):
	n = W.shape[0]
	
	L = csgraph.laplacian(W, normed=False)
	
	constrained_nodes = X0 + X1
	unconstrained_nodes = [i for i in range(n) if i not in constrained_nodes]
	
	b = np.zeros(n)
	for x1 in X1:
		b[x1] = 1
	
	# print(L)
	# print(b)

	L_unconstrained = L[np.ix_(unconstrained_nodes, unconstrained_nodes)]															# Gets diagonal subgraph of L
	# print(L_unconstrained)

	b_unconstrained = b[unconstrained_nodes] - np.matmul(L[np.ix_(unconstrained_nodes, constrained_nodes)], b[constrained_nodes])	# B with the subtraction of the constrained term
	# print(b_unconstrained)
	
	v_unconstrained = solve(L_unconstrained, b_unconstrained)
	
	voltages = np.zeros(n)
	voltages[X0] = 0
	voltages[X1] = 1
	voltages[unconstrained_nodes] = v_unconstrained
	
	return voltages

def addGround(data, W):
	n = W.shape[0]
	newW = np.zeros([n + 1, n + 1])

	newW[0:n,0:n] = W

	p_g = 0.01

	for x in range(0, n):			# W[g, g] = 0
		newW[x][n] = p_g / n
		newW[n][x] = p_g / n

	return newW

def radialkernel(x, y, r):
	if (distance(x, y) < r):
		return 1
	else:
		return 0

def gaussiankernel(x, y, std):
	return np.exp(np.pow(distance(x, y) / std, 2) / -2)

def getWeights(data, kernel, c):								# Both need some constant parameter C
	n = len(data)

	weights = np.zeros([n, n])
	for x in range(0, n):
		for y in range(x, n):
			v = kernel(data[x], data[y], c) / np.pow(n, 2)		# R = n^2/k, W = 1/R = k/n^2

			weights[x][y] = v
			weights[y][x] = v

	return weights

# Example usage
if __name__ == "__main__":
	data = load_data_json("line.json")
	n = len(data)

	X0 = [x for x in range(n) if data[x][0] < 1]
	X1 = [x for x in range(n) if data[x][0] > 2]

	W = getWeights(data, kernel=gaussiankernel, c=0.03)
	voltages = compute_voltages(W, X0, X1)

	W = getWeights(data, kernel=gaussiankernel, c=0.3)
	grounded_voltage = compute_voltages(addGround(data, W), [n], X1)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter([x[0] for x in data], voltages, c='r', marker='o', label='Voltage Points')
	ax.scatter([x[0] for x in data], grounded_voltage[:-1], c='b', marker='.', label='Grounded Points')
	ax.legend()
	plt.show()