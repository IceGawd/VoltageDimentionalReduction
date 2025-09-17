"""
Streaming K-Means++ Implementation with FAISS

This module implements a memory-efficient streaming version of the k-means++
clustering algorithm using FAISS for fast nearest neighbor search. It is designed
to handle large datasets that don't fit in memory by processing data in batches.

Key Features:
	- Streaming processing for memory efficiency
	- FAISS-based distance computations for speed
	- Automatic normalization constant estimation
	- Support for normalized and unnormalized vectors
	- Label tracking for supervised applications

Example:
	>>> from kmeans import Streaming_Kmeans
	>>> centroids, counters, label_counts, init_d2, final_d2 = Streaming_Kmeans('data.csv')
	>>> print(f"Found {len(centroids)} clusters")
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional, Counter as CounterType
import numpy.typing as npt

from Utilities import config
from Visualization import generalVisualization
from Utilities.reader import Reader

import faulthandler
from collections import Counter
faulthandler.enable()

# ------------- Streaming KMeans++ --------------
class StreamingKMeansPlusPlus:
	"""
	Implements streaming k-means++ centroid selection using FAISS for efficient distance computation.

	This class implements a streaming version of the k-means++ initialization algorithm.
	It processes data in batches and maintains a FAISS index for efficient nearest
	neighbor searches. New centroids are selected probabilistically based on their
	squared distances to existing centroids.

	Attributes:
		d (int): Dimensionality of the input vectors.
		Z (float): Scaling constant for sampling probability (max_dist² - min_dist²).
		shift (float): Minimum squared distance for probability calculation.
		max_centroids (int): Maximum number of centroids to maintain.
		index (faiss.IndexFlatL2): FAISS index for efficient distance computation.

	Note:
		- The algorithm maintains normalized probabilities using Z and shift parameters
		- FAISS index must be of type IndexFlatL2 for correct distance calculations
		- Vectors can optionally be normalized before processing
	"""

	def __init__(self, d: int, max_dist2: float, min_dist2: float, index: faiss.IndexFlatL2) -> None:
		"""
		Initialize the streaming k-means++ algorithm.

		Args:
			d (int): Dimensionality of the input vectors.
			max_dist2 (float): Maximum squared distance between any two points in initial buffer.
			min_dist2 (float): Minimum squared distance between any two points in initial buffer.
			index (faiss.IndexFlatL2): Pre-initialized FAISS index for distance computations.

		Raises:
			AssertionError: If the provided index is not of type faiss.IndexFlatL2.

		Note:
			- Z is computed as (max_dist² - min_dist²) for probability normalization
			- The shift parameter ensures non-negative distance values
			- max_centroids is loaded from the global config
		"""
		self.d = d
		self.Z = max_dist2 - min_dist2  # Normalization constant for sampling probabilities
		self.shift= min_dist2  # Shift to ensure non-negative distances
	
		self.max_centroids = config.params['max_centroids']
		assert type(index) == faiss.IndexFlatL2, "Index must be of type faiss.IndexFlatL2"
		self.index = index
	 


	def _compute_distances_squared(self, X, index):
		"""
		Computes squared distances from X to nearest centroid in index.

		Args:
			X (np.ndarray): Batch of input vectors.
			index (faiss.IndexFlatL2): FAISS index of centroids.

		Returns:
			D: Squared distances for each point in X to the nearest centroid.
			I: Indices of the nearest centroids in the index.
		"""
		if self.index is None or self.index.ntotal == 0:
			print("Index is empty, returning infinity distances.")
			return None
		D, I = self.index.search(X, k=1) # D[i, 0] is the squared distance from X[i] to the nearest centroid
		return D[:, 0], I[:, 0]  # Return only the squared distances and indices of nearest centroids   

	def update(self, X_batch):
		"""
		Add a centroid to the list via probabilistic sampling. (Similar to KMeans++)

		Args:
			X_batch (np.ndarray): Normalized batch of vectors.
		"""
				
		needed_centroids = self.max_centroids - self.index.ntotal
		
		d2,I = self._compute_distances_squared(X_batch, self.index)
		if needed_centroids>0:
			ratio=(d2-self.shift) / self.Z  # Shift distances 
			probs = np.minimum(np.maximum(ratio, 0), 1)  # Ensure probabilities are in [0, 1]
			#print('probs=',np.mean(probs))
			rand_vals = np.random.rand(X_batch.shape[0])
			accept_mask = (rand_vals < probs) & (d2 > 0)
			if np.sum(accept_mask) != 0:
				X=np.stack(X_batch[accept_mask])
				if self.index.ntotal + X.shape[0] > self.max_centroids:
					X = X[:self.max_centroids - self.index.ntotal]
				if config.params['normalize_vecs']:
					X = X / np.linalg.norm(X,axis=1, keepdims=True)
				self.index.add(X)  # Add new vectors to the index          
				print(f"\nnumber of centroids: {self.index.ntotal}, max_centroids: {self.max_centroids}")
		
	def get_centroids(self):
		"""
		Returns the current list of centroids as a NumPy array.

		Returns:
			np.ndarray: Centroids of shape (num_centroids, d)
		"""
		return self.index.reconstruct_n(0, self.index.ntotal)



# ------------------- Streaming_Kmeans----------
def Streaming_Kmeans(filepath: str) -> Tuple[npt.NDArray, npt.NDArray, List[Optional[CounterType]], float, float]:
	"""
	Perform streaming k-means++ clustering with FAISS.

	This function implements a complete streaming k-means++ pipeline, including initialization
	and refinement. It processes data in batches to handle large datasets efficiently.

	Algorithm Steps:
		1. Initialize: Read initial buffer to estimate distance parameters
			- Compute pairwise distances using FAISS
			- Determine max_dist² and min_dist² for probability scaling
		2. Select Centroids: Stream data to select initial centroids
			- Use distance-based probabilistic sampling
			- Continue until max_centroids is reached
		3. Refine Centroids: Stream data twice to update centroids
			- First pass: Assign points and update centroids
			- Second pass: Further refine centroids
			- Track label distributions for each centroid

	Args:
		filepath (str): Path to the input data file containing vectors.

	Returns:
		Tuple containing:
			- centroids (np.ndarray): Final centroid vectors (shape: [n_centroids, d])
			- counters (np.ndarray): Number of points assigned to each centroid
			- label_counts (List[Counter]): Distribution of labels for each centroid
			- initial_mean_d2 (float): Initial mean squared distance to centroids
			- mean_d2 (float): Final mean squared distance to centroids

	Note:
		Configuration parameters are read from config.params:
			- 'init_size': Size of initial buffer for parameter estimation
			- 'batch_size': Number of vectors to process per batch
			- 'max_centroids': Maximum number of centroids to select
			- 'normalize_vecs': Whether to normalize input vectors
	"""
	reader = Reader(filepath)

	# Step 1: Read initial buffer of vectors for Z estimation
	######################################

	buffer = []
	d = None
	total_needed = config.params['init_size']
	collected = 0
	for vectors, _ in reader.stream_batches(config.params['batch_size']):
		if d is None:
			d = vectors.shape[1]
		if collected + len(vectors) > total_needed:
			vectors = vectors[:total_needed - collected]
		buffer.append(vectors)
		collected += len(vectors)
		if collected >= total_needed:
			break

	buffer = np.vstack(buffer)
	if config.params['normalize_vecs']:
		buffer = buffer / np.linalg.norm(buffer, axis=1, keepdims=True)

	# Compute all pairwise distances using FAISS and set Z to the maximal distance
	index = faiss.IndexFlatL2(d)  # this should be the one and only place that FAISS in initialized, otherwise there are problems with 
								  # Initializing openMP more than once
	index.add(buffer)
	D, _ = index.search(buffer, buffer.shape[0])  # D[i, j] is the squared L2 distance from buffer[i] to buffer[j]
	np.fill_diagonal(D, -np.inf)
	max_dist2 = np.max(D[d>0])  # max of square distances
	min_dist2 = np.min(D[D > 0])  # Minimum distance in the buffer (excluding self-distances)
	print(f"\nEstimated max pairwise distance squared, FAISS) = {max_dist2:.4f}")
	print(f" Estimated minimum distance squared = {min_dist2:.4f}")

	# Step 2: Streaming centroid selection using kmeans++ like rule
	######################################
	# For seeding, pick a random vector from the buffer
	centroids = buffer[:10,:]

	reached_max_centroids = False
	while not reached_max_centroids:
		print(f"\nStarting a new attempt to select up to {config.params['max_centroids']} centroids")
		index.reset()  # Reset index to ensure it's empty
		index.add(centroids)  # Add the initial centroid to the index
		print(index.ntotal, "vectors in index after adding initial centroid")
		skmeans = StreamingKMeansPlusPlus(d=d, max_dist2=max_dist2, min_dist2=min_dist2,index=index)    

		for vectors, _ in reader.stream_batches(config.params['batch_size']):
			if config.params['normalize_vecs']:
				vectors = vectors/ np.linalg.norm(vectors, axis=1, keepdims=True)
			
			skmeans.update(vectors)
			if(index.ntotal >= config.params['max_centroids']):
				print(f"Reached maximum number of centroids: {index.ntotal}")
				reached_max_centroids=True
				break
		if not reached_max_centroids:
			if reader.get_counter() < config.params['batch_size']*2:
				raise ValueError(f"Only reached {index.ntotal} and {reader.get_counter()}<{config.params['batch_size']*2}, not enough data to continue, exiting")
			print(f"Only reached {index.ntotal} centroids, restarting with a smaller Z")
			max_dist2 -= (max_dist2 - min_dist2) * 0.5  # Reduce max_dist2 to increase sampling probability
			print(f"New max_dist2={max_dist2}, min_dist2={min_dist2}")
			reader.close()
			reader = Reader(filepath)  # reopen the data file



	# Step 3. update centroids using a streaming version of the Kmeans algorithm
	######################################
	centroids = skmeans.get_centroids()  #extract centroids from the faiss index
	counters=np.ones(centroids.shape[0], dtype=np.int32)  # Initialize counters for each centroid
	
	total_count = 0  # Total number of vectors processed
	 # compute for each centroid a label that is the majority of examples that are assigned to it

############################ begin refine_centroids
	def refine_centroids(centroids, vectors, other):
		""" Refine centroids using the current batch of vectors
		"""

		_, vec_assignments = skmeans._compute_distances_squared(vectors, skmeans.index)
		# Assign vectors and labels to centroids

		# Create a dictionary to hold centroid statistics
		# Each entry will contain:
		# - 'centroid': current centroid vector
		# - 'vectors': list of vectors assigned to this centroid
		# - 'label': list of labels of the assigned vectors
		# - 'other': list of other fields of the assigned vectors
		centroids = skmeans.get_centroids()  # Get the current centroids from the index
		centroid_stats = {i:{'centroid':centroids[i], 'vectors': [], 'label': [], 'other': []} for i in range(centroids.shape[0])}

		for idx, centroid_idx in enumerate(vec_assignments):
			centroid_stats[centroid_idx]['label'].append(other[idx]['label'])
			centroid_stats[centroid_idx]['vectors'].append(vectors[idx])
			centroid_stats[centroid_idx]['other'].append(
				{k: v for k, v in other[idx].items() if k != 'label'}
			)

		# Compute RMS error and updated mid-point for each centroid
		rms=0
		alpha=config.params['kmeans_alpha']
		for i in centroid_stats:
			if len(centroid_stats[i]['vectors']) > 0:
				rms += np.mean(np.square(centroid_stats[i]['vectors'] - centroid_stats[i]['centroid']))
				old_centroid = centroid_stats[i]['centroid']
				vectors_mean = np.mean(centroid_stats[i]['vectors'], axis=0)
				new_centroid= vectors_mean*alpha + old_centroid*(1-alpha)
				centroid_stats[i]['centroid'] = new_centroid
		rms = np.sqrt(rms / len(centroid_stats))

		too_small=[]
		# collect information about too large and too small centroids
		for i in range(len(centroid_stats)):
			if len(centroid_stats[i]['vectors']) ==0:
				too_small.append(i)

		sizes = np.array([len(centroid_stats[i]['vectors']) for i in centroid_stats])
		order = np.argsort(sizes)[::-1]  # Sort indices by size in descending order
		sorted_sizes = np.sort(sizes)[::-1]  # Sorted sizes in descending order
		if(config.params['verbosity']>=2):	
			print(f"sorted_sizes={sorted_sizes[:5]}")  # Print the top 5 sizes for debugging
			print(f"too_small={too_small}")
		new_centroids = np.array([centroid_stats[i]['centroid'] for i in centroid_stats])
		if config.params['normalize_vecs']:
			centroids=centroids/ np.linalg.norm(centroids, axis=1, keepdims=True)

		# Update faiss index with new centroids
		
		skmeans.index.reset()  # Reset index to ensure it's empty
		skmeans.index.add(new_centroids)  # Add the final centroids to the index
		return centroid_stats, rms,len(too_small)
			
########################### end refine_centroids

	for vectors, other in reader.stream_batches(config.params['batch_size']):
		if len(vectors) < config.params['batch_size']:
			break
		if config.params['normalize_vecs']:
			vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

		# Refine centroids using the current batch of vectors
		# This function will return the updated index and centroid statistics
		centroid_stats, rms, len_too_small  = refine_centroids(centroids, vectors, other)

		print(f"too_small={len_too_small}, rms={rms}")

	
	# Compute majority label for each centroid
	majority_labels = []
	label_counters =  []
	centroid_others =  []
	for i in centroid_stats.keys():
		labels_list = centroid_stats[i]['label']
		other_list = centroid_stats[i]['other']
		if labels_list:
			majority_label = Counter(labels_list).most_common(1)[0][0]
			label_counter = Counter(labels_list)
		else:
			majority_label = None
			label_counter = None
		majority_labels.append(majority_label)
		label_counters.append(label_counter)
		centroid_others.append(other_list)
	

	# Close the reader 
	print("\nClosing reader...")
	reader.close()

	return centroids, counters, majority_labels, label_counters, centroid_others, rms


# ------------------- Main ---------------------
from Utilities.set_params import set_params

def main() -> None:
	"""
	Main entry point for the streaming k-means++ clustering algorithm.

	This function:
		1. Sets up configuration parameters from command line arguments
		2. Runs the clustering algorithm on the specified input file
		3. Handles test mode with synthetic data
		4. Saves results and generates visualizations if requested

	Configuration (via config.params):
		- test: Boolean, enables test mode with synthetic data
		- file_path: Path to input data file
		- split_char: Delimiter for CSV files
		- normalize_vecs: Whether to normalize vectors
		- max_centroids: Maximum number of centroids
		- init_size: Size of initial buffer
		- batch_size: Number of vectors per batch
		- output: Path for saving results (optional)
		- verbosity: Level of output detail

	In test mode:
		- Uses synthetic 2D random data
		- Validates mean squared distance
		- Generates visualization if successful
	"""
	
	set_params()  # Set parameters according to the command line           
	if config.params['test']:
 
	  config.params['file_path']= '../../Voltage_Data/synthetic/2drandom10000.csv'
	  config.params['split_char']= ','
	  config.params['normalize_vecs']= False

	  config.params['max_centroids']= 20
	  config.params['init_size']= 1000
	  config.params['batch_size']= 100
	  config.params['output']=None

	centroids,counters,majority_labels,label_counters,rms=Streaming_Kmeans(config.params['file_path'])

	# Finalization and saving
	print(f"\nNumber of centroids in index after finalization: {centroids.shape[0]}")
	print('Final mean squared distance:', rms)
	if config.params['test']:
		if rms>0.04 or rms<0.035:
			raise ValueError(f"test failed, rms={rms} is outside the range [0.035,0.04]")
		else:
			print('Test Passed')
	if config.params['output'] is not None:
		np.savez(config.params['output'], centroids=centroids, counters=counters, label_counts=label_counts,)
		print(f"Centroids saved to {config.params['output']}")
	else:
		print("No output file specified, centroids not saved.")

	# if 2d test then visualize datapoints, centroids labels
	if config.params['test']:
		generalVisualization.plot_centroids(centroids, counters, label_counts)
	
if __name__ == "__main__":
	main()

