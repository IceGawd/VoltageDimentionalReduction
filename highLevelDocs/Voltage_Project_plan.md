# Description
This project is by Yoav Freund's Lab: yfreund@ucsd.edu

The goal of this project is to create an efficient implementation of
the voltage-maps idea as described in the paper "Structure from
Voltage" by Robi Bhattacharjee , Alex Cloninger , Yoav Freund and
Andreas Oslandsbotn available from Arxiv.

Another paper that is important for understanding this project is 
Doyle and Snell "Random walks and electric networks"

The Structure from Voltage paper proposes an algorithm for analyzing
a data cloud in a metric space and to
use this map to generate two dimensional visualizations of parts of
this data cloud. The work is based on the seminal work of Belkin and
Nyogi "Laplacian Eigenmaps for Dimensionality Reduction and Data Representation"

Both diffusion maps and structure from voltage can be described using
the concept of resistor graphs, voltages and currents as described in
Doyle and Snell. The main difference between diffusion maps and
structure from Voltage is that the functions used in the first are
eigen-functions of the Laplace operator which characterize the decay
of voltage functions to the zero-everywhere voltage, while the vectors
use in the second describe the voltage function where one node (or set
of nodes) called a *landmark* is set to voltage one and and a
universal ground node is set to voltage zero.

The solution is a voltage function that is equal to 1 on the landmark
and decays to zero exponentially with the distance from the
landmark. The advantage of this exponential decay is that solving the
voltage function can ignore the points that are far from the
landmark. This allows distributing the computation of the voltage
functions to different cores.

Another, and probably more important method for making the computation
fast is to use, instead of the raw data, a spatial partition of the
data cloud into cells, replacing individual datapoints with centroids
weighted by the number of data-points. The current idea is to use
k-means to generate the partition. Relying on **streaming** K-means++
and K-means allows fast processing of large files that don't fit in memory.

## Skills required

* Experience in data analysis
* Curiosity and drive.
* Read the first 23 pages of Doyle and Snell, solved the problems up to 1.2.8 and is able to discuss the text and explain the solutions.
* Writing well organized, well documented efficient and readable python code.
* Numpy, Pandas, Scipy. Array operations rather than for loops.
* jupyter notebooks.
* GitHub, including creating and merging branches.
* Python documentation as described in [pydoc](https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)


