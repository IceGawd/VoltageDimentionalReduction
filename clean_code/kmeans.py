import numpy as np
import faiss

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

    Attributes:
        d (int): Dimensionality of vectors.
        Z (float): Scaling constant for sampling probability.
        max_centroids (int): Maximum number of centroids to retain.
    """

    def __init__(self, d, max_dist2,min_dist2,index):
        """
        Initializes the streaming k-means++ class.

        Args:
            d (int): Vector dimensionality.
            Z (float): Normalization constant for sampling.
            max_centroids (int): Maximum number of centroids to store.
            index (faiss.IndexFlatL2): FAISS index for efficient distance computation.
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
def Streaming_Kmeans(filepath):

    """
    Main function to perform streaming k-means++ with FAISS.

    Steps:
        1. Estimate normalization constant Z from an initial buffer.
        2. Select centroids incrementally using streaming batches.
        3. update centroids using a streaming version of the Kmeans algorithm
        4. Save final centroids to a .npy file.

    parameters are passed through config.params, see listing of parameters in argparse section.
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
        buffer= buffer / np.linalg.norm(buffer, axis=1, keepdims=True)

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
    index.reset()  # Reset index to ensure it's empty
    index.add(centroids)  # Add the initial centroid to the index
    print(index.ntotal, "vectors in index after adding initial centroid")
    skmeans = StreamingKMeansPlusPlus(d=d, max_dist2=max_dist2, min_dist2=min_dist2,index=index)
    
    for vectors, _ in reader.stream_batches(config.params['batch_size']):
        if config.params['normalize_vecs']:
            vectors=vectors/ np.linalg.norm(vectors, axis=1, keepdims=True)
        
        skmeans.update(vectors)
        if(index.ntotal >= config.params['max_centroids']):
            print(f"Reached maximum number of centroids: {index.ntotal}")
            break

    # Step 3. update centroids using a streaming version of the Kmeans algorithm
    ######################################
    centroids = skmeans.get_centroids()  #extract centroids from the faiss index
    counters=np.ones(centroids.shape[0], dtype=np.int32)  # Initialize counters for each centroid
    
    total_count = 0  # Total number of vectors processed
     # compute for each centroid a label that is the majority of examples that are assigned to it


    def refine_centroids(centroids, vectors,labels):
        """ Refine centroids using the current batch of vectors. This is done based on the observation that with the current update rule 
        some centroids are assigned many vectors and some none. The idea here is to eliminate those that are assigned too few and on the other hand split those
        that are assigned too many vectors.
        """

        _, vec_assignments = skmeans._compute_distances_squared(vectors, skmeans.index)
        # Assign vectors and labels to centroids

        # Create a dictionary to hold centroid statistics
        centroids = skmeans.get_centroids()  # Get the current centroids from the index
        centroid_stats = {i:{'centroid':centroids[i], 'vectors': [], 'labels': []} for i in range(centroids.shape[0])}

        for idx, centroid_idx in enumerate(vec_assignments):
            centroid_stats[centroid_idx]['labels'].append(labels[idx])
            centroid_stats[centroid_idx]['vectors'].append(vectors[idx])

        # Compute RMS error and updated mid-point for each centroid
        rms=0
        alpha=config.params['alpha']
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
        print(f"sorted_sizes={sorted_sizes[:5]}")  # Print the top 5 sizes for debugging
        if config.params['equalize_centroids']:
            ## Remove centroids that are too small
            for j in range(len(too_small)):
                i1 = too_small[j]   #points to the j'th small centroid
                i2 = order[j] # points to the j'th largest centroid
                #define two centroids that spit the largest centroid
                #print(f"{len(centroid_stats[i2]['vectors'])} vectors assigned to centroid {i2}")

                # perturb the centroid vector to create a new centroid
                v = centroid_stats[i2]['centroid']  # Current centroid vector
                d = v.shape[0]  # Dimensionality of the centroid
                e = np.random.randn(d)
                e = e * 1e-20 / np.linalg.norm(e)  # Normalize to unit length

                centroid_stats[i1]['centroid'] = v+e


        new_centroids = np.array([centroid_stats[i]['centroid'] for i in centroid_stats])
        if config.params['normalize_vecs']:
            centroids=centroids/ np.linalg.norm(centroids, axis=1, keepdims=True)

        # Update faiss index with new centroids
        
        skmeans.index.reset()  # Reset index to ensure it's empty
        skmeans.index.add(new_centroids)  # Add the final centroids to the index
        finished=not config.params['equalize_centroids'] or len(too_small)==0
        return centroid_stats, rms,len(too_small),finished

    
        
    for vectors, labels in reader.stream_batches(config.params['batch_size']):
        if len(vectors)<config.params['batch_size']:
            break
        if config.params['normalize_vecs']:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        finished = False
        while not finished:
            # Refine centroids using the current batch of vectors
            # This function will return the updated index and centroid statistics
            print(f"\nrefining vectors")
            centroid_stats, rms, len_too_small, finished = refine_centroids(centroids, vectors, labels)
            print(f"too_small={len_too_small}, rms={rms}, finished={finished}")

    
    # Compute majority label for each centroid
    majority_labels = []
    label_counters =  []
    for i in centroid_stats.keys():
        labels_list = centroid_stats[i]['labels']
        if labels_list:
            majority_label = Counter(labels_list).most_common(1)[0][0]
            label_counter = Counter(labels_list)
        else:
            majority_label = None
            label_counter = None
        majority_labels.append(majority_label)
        label_counters.append(label_counter)
    

    # Close the reader 
    print("\nClosing reader...")
    reader.close()

    return centroids, counters, majority_labels, label_counters, rms


# ------------------- Main ---------------------
from Utilities.set_params import set_params

def main():
    
    set_params()  #set parameters accordinig to the command line            
        
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

