### Using XGBoost to predict the labels from the voltage

Terms:
* Data-point: the lowest level instance
* Centroids: the centers of cells
* Landmarks: locations where the voltage is set to 1.

*Yoav* you don't need to do all of the steps until my next comment.

-Load the full dataset using the Reader class. 
-Run the Streaming_Kmeans() function on the full dataset to identify a set of centroids. These centroids act as candidate landmarks. Each centroid is returned along with its associated counter (how many points it attracted) and majority class label (for reference).
-Select which centroids to be chosen as  landmarks - optimized landmark selection is in process.
-Convert each selected centroid into a Landmark object, assigning a voltage of 1 to each. These will serve as "sources" in the voltage computation step.
-Construct a Problem object using all data points (SetOfPoints) and a fixed ground resistance value r(which is still in process). This sets up the resistance graph for the entire dataset.
-Create a Solver object that takes the problem and will be used to compute the voltage response across the graph.
-Initialize a VoltageMap to store the voltage vector for each *Centroid*. Each vector contains the voltage associated with one of the landmarks.

*Yoav*: All of the steps above you can do by running main.py without parameters. The resulting voltage_map is stored in a pkl file that you can read into your program.

-For each landmark:
    -Use the Solver.compute_voltages() function to compute the voltage values for all points.
    -Add the resulting vector to the VoltageMap. After this loop, we will have one voltage vector per landmark, for every data point.
-Stack the voltage vectors to form a matrix of size (n_points, n_landmarks). Each row represents a transformed feature vector for a data point, where each entry corresponds to the response to a specific landmark.

*Yoav* This would not work because compute_voltages requires all of the points to be in memory. The solution we talked about is that each data point is processed separately, using the voltage vectors associated with 
it's nearest K centroids.


## Running CGBoost
-Split the voltagemap into training and test sets using an 80-20 split:
(We can choose to shuffle the data here)
-Use the 80% of rows as the training set (X_train, y_train)
-Use the remaining 20% as the test set (X_test, y_test)
-Train an XGBoost classifier on the voltage-based training features and their corresponding labels.
