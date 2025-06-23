## Minor changes
* Problem.optimize and compute_voltages should work on a single landmark at a time.
* Problem set_matrix uses a self.matrix to compute the matrix just once
* In general minimize number of parameters, move shared parameters to config.params
* Avignha to write directions for how to create documentation. Documentation.md
* Put data files on Google drive
* 
## Testing
Testing should give a pass/fail and not require human judgement. (Other than visualization)
* For Kmeans - test based on average distance to centroid.
* For Solver - test using a random walk simulator as a comparison
* Shivani - to go over python files on clean_code and write tests.

## ongoing
* Avighna works on improving visualization
* Yoav works on selecting landmarks.


## Characterizing the neighborhood of landmarks 
We want to choose the resistance to the ground so that it would capture low dimensional structure.
In other words, we want to find the approximate the intrinsic dimension of the neighborhood. A dimension of zero corresponds to a cluster that is almost isolated from the rest of the graph.

To do this estimation I suggest looking at the relationship between the average potential over the whole graph as a function of the resistance to the ground.

## Evaluating solutions
* We need a way to judge the quality of voltage maps without using
  visualization, so that we can separate out the effects of the
  visualization. Here are two suggestions: 
  1. use XGBoost to predict the labels from the voltage. I.e. use each voltage function as a feature.
  1. A ssociate with each
  centroid the distribution of labels of the
  points associated with the centroid. Compute the correlation
  between the vector of voltages and these
  distributions.
