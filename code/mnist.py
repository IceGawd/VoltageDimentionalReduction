import create_data
import kmeans
import voltage
import os
import importlib
import time
import bpf
import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

print("Loading Data...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(np.int64)

# data = create_data.Data(np.array(X))
# MNIST Pre-processing

subDivision = {}
summation = {}
count = {}

print("Sorting and averaging...")

for xi, yi in zip(X, y):
    if yi in summation:
        subDivision[yi].append(np.array(xi))
        summation[yi] += np.array(xi)
        count[yi] += 1
    else:
        subDivision[yi] = [np.array(xi)]
        summation[yi] = np.array(xi)
        count[yi] = 1

# Kmeans sampling equal points from each of the 10 digits

print("Kmeans...")

points = 1000
k = points // 10
data = []

for yi in range(10):
    print(yi)
    
    partitions = kmeans.Partitions(subDivision[yi])
    partitions.k_means(k, seed=time.time())

    data += list(partitions.centers)


# Create the landmarks

data = create_data.Data(data)

landmarks = []
for yi in range(10):
    landmark = voltage.Landmark(-1, 1)

    ignore = []
    while (landmark.index // k != yi):
        ignore.append(landmark.index)

        landmark = voltage.Landmark.createLandmarkClosestTo(data, summation[yi] / count[yi], 1, ignore=ignore)
 
    landmarks.append(landmark)

print(len(data))
print([l.index for l in landmarks])


print(type(data))
print(isinstance(data, create_data.Data))


print("Parameter Finding...")

cs = []
pgs = []

param_finder = bpf.BestParameterFinder()

"""
for landmark in landmarks:
    c, p_g = param_finder.bestParameterFinder([landmark], data, minBound=-10, maxBound=20, granularity=3, epsilon=0.5, approx=10)
    print(c, p_g)
    cs.append(c)
    pgs.append(p_g)

print(cs)
print(pgs)
"""

c, p_g = param_finder.bestParameterFinder(landmarks, data, minBound=-10, maxBound=10, granularity=3, epsilon=0.5, approx=10)

voltages = []

for index in range(0, len(landmarks)):
    problem = voltage.Problem(data)
    problem.setKernel(problem.gaussiankernel)
    problem.setWeights(c)
    problem.addLandmark(landmarks[index])
    problem.addUniversalGround(p_g)
    voltages.append(voltage.Solver(problem).approximate_voltages(max_iters=10))


param_finder.visualizations(voltages, "../inputoutput/matplotfigures/MNIST")

print(voltages[0])
print(voltages[1])

predicted = np.argmax(voltages, axis=0)
correct = np.repeat(np.arange(10), k)

num_incorrect = np.sum(predicted != correct)

accuracy = np.mean(predicted == correct)
error_rate = 1 - accuracy

print(predicted)
print(f"Incorrect predictions: {num_incorrect}/100")
print(f"Accuracy: {accuracy:.2%}")
print(f"Error Rate: {error_rate:.2%}")