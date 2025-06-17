

* To implement greedy search we need to add one landmark at a
  time. Which means that Problem.optimize and compute_voltages should
  work on a single landmark at a time.
  
* Given a set of landmarks and their corresponding voltage maps. We
  should consider the distribution of the max voltage over the
  centroids. We should choose the next node to be a random one of the
  x% lowesst percentile.

* We need a way to judge the quality of voltage maps without using
  visualization, so that we can separate out the effects of the
  visualization. Here is a suggested way: we associate with each
  centroid a distribution, probably the distribution of labels of the
  points associated with the centroid. We then compute the mutual
  information between the vector of the voltages and these
  distributions.
