"""The current implementation to follow the new project plan

setofpoints.py: two numpy arrays points: an array of dimension nXd where n  is the number of points, d is the dimension of each point. weights, an array of size n which defines the weights of the points.
problem.py: Contains a set of points, a kernel width c and a resistance to the ground r. The class has a method: calcResisitanceMatrix which outputs a (potentially sparse) matrix of size (n+1)X(n+1) which defines the weights associated with pairs of points. The matrix can be sparse if we zero the weights are smaller than some threshold
solver.py: Finds the voltage map for each point in a pointset for a given problem for a given landmark (a point or set of points)
kmeans.py:  given a data file, uses streaming to generate a weighted set of points (a partition).
visualization.py: create 2D visualizations of the voltage map
map.py: a collection of solutions which defines the location of each point
"""
