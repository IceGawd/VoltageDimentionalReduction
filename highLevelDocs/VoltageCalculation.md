# Resistance Network Voltage Solver Documentation

## Overview

This document describes the mathematical formulation and implementation of a voltage solver over a resistance network. The core idea is to model a graph of points where some nodes called "landmarks" are held at fixed voltages. The remaining unconstrained node voltages are computed by solving a linear system that follows Kirchoff's Law.

## Problem Setup

First thing that is done is the calculation of the resistance matrix. This name is a bit misleading, as we actually generate the matrix which we will calculate the matrix inverse of when solving for the voltages. The matrix we are inverting is $I - P$ where $P$ is the probability of which node to go to if we view this as a random walk. This probability is called a connectivity matrix, which is a resistance matrix but each value is inverted. Since two unconnected nodes have an infinite resistance, it is easier to represent it as its inverse which is $\frac{1}{\infty} = 0$. We do $A = I - P$ because the equation we intend to solve is $Ax = b$. Recall that Kirkchoff's Law states that the sum of currents entering a node has to be the same as the sum of currents leaving the node. In our case, each "node" is a data point or kmeans center of a data point. So, for a node $x_i$, we are solving $x_i - \sum_{}^{} \frac{x_j}{k} = b_i$ where each $x_j$ is a neighboring node and $b_i$ is either $0$ if not directly connected to the voltage source and $-\frac{1}{k}$ if connected to the voltage source. 

## Terminology

- **node**: A node could either be a k-means center or a data point itself. Either way its irrelevant and dealt with in SetOfPoints
- **kernel**: This is the weighted connections for k-nearest neighbors
- **weights**: This is the probability matrix; for each element $i, j$, the value represents the probability of going from point $i$ to point $j$. Includes ground
- **voltages**: This is the array that respresents all of the voltages
- $\mathbf{A}$: This is the matrix which represents the linear system of equations to calculate the voltage at each node
- $\mathbf{x}$: This is a vector that represents the voltages that we solve for, only the unconstrained ones
- $\mathbf{b}$: This is a vector that 


## calcResistanceMatrix

First thing we do is calculate the nearest neighbors used to get which nodes are connected and which are not. We choose k + 1 since one of the points will be the queried point itself, and we will remove in the creation of the kernel.

```
nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
_, indices = nbrs.kneighbors(X)
```

Next we create the base kernel. This kernel specifies the connectivity between all the nodes, not including the ground node itself. We row as each row represents the probability of going to the next node. We normalize before adding the ground so that each node has the same probability of going to the ground rather than each node having a different probability of going to the next node.

```
kernel = np.zeros((n, n), dtype=float)
weight = 1.0 / k

for i in range(n):
	for j in indices[i]:
		if j != i:
			kernel[i, j] = weight * self.points.weights[i] * self.points.weights[j]
			kernel[j, i] = weight * self.points.weights[j] * self.points.weights[i]


kernel = kernel / kernel.sum(axis=1, keepdims=True)
```

To give meaning to ``self.r``, we define connectivity by scaling the resistance by the number of nodes and then taking the inverse. The purpose is because the connectivity to the ground essentially represents the probability of a random walk going to the ground. Thus, the average amount of nodes a random walker would go before going to the ground is $\frac{1}{\frac{1}{self.r * n}} = self.r * n$. In other words, ``self.r`` represents the percent of how many nodes does a random walk electron go before reaching the ground. Do note that a random walker can walk on a node multiple times so ``self.r`` being 1 does not guarantee that all nodes will have a voltage of 1, however they will most likely be very close to 1.

We add a ground row so that we return a square matrix, though the contents of it are irrelevant since it gets removed in compute_voltages since the ground is a sink.

```
connectivity = 1 / (self.r * n)
ground_col = np.full((n, 1), connectivity, dtype=float)
ground_row = ground_col.T

top    = np.hstack((kernel, ground_col))
bottom = np.hstack((ground_row, [[0]]))
full   = np.vstack((top, bottom))
```

Finally, we have to re-normalize so that all rows add up to 1 for the weights. We then take this and subtract it from the identity matrix to get the matrix we will use in compute_voltages. [In the problem setup](#problem-setup) it is explained why we do this.

```
row_sums = full.sum(axis=1, keepdims=True)
weights = full / row_sums
return np.identity(weights.shape[0]) - weights
```

## compute_voltages

Everything done in **calcResistanceMatrix** is all the setup that has to be done before we get the landmark. This doesn't mean that the matrix calculated is the matrix we will use to get $x = A^{-1}b$. First we need to locate all of the constrained nodes, which in the specific uses of this project are just the ground and chosen landmark, and then we remove them. We do this because we don't need to calculate voltages for nodes with a set voltage. We then calculate the b vector [as explained here](#problem-setup).

```
b = np.zeros(n)
for lm in landmarks:
	for y in range(0, n):
		b[y] -= lm.voltage * weights[y][lm.index]

A_unconstrained = weights[np.ix_(unconstrained_nodes, unconstrained_nodes)]
b_unconstrained = b[unconstrained_nodes]
```

Now that we have the $A$ matrxi and $b$ vector, we can do a matrix solve and then remove the ground (since the ground was not in the original dataset).

```
v_unconstrained = solve(A_unconstrained, b_unconstrained)

self.voltages = np.zeros(n)

for lm in landmarks:
	self.voltages[lm.index] = lm.voltage

self.voltages[unconstrained_nodes] = v_unconstrained

self.voltages = self.voltages[:-1]
```