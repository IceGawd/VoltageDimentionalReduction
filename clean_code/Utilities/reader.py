import numpy as np
import setofpoints
from Utilities import config
from Utilities.set_params import set_params

class ParseException(Exception):
	"""
	Exception raised when parsing vector data from text files fails.
	
	This exception is raised when a line in the input file cannot be parsed
	into the expected format of label followed by numerical values.
	"""
	pass

def readvec(file, is_binary=False):
	"""
	Read a single vector and its label from a text file.
	
	Parses one line from the input file and extracts the label and vector data.
	Lines starting with '#' are treated as comments and ignored. The function
	expects each line to have at least 2 parts: a label and one or more numerical values.
	
	Args:
		file (TextIO): Open file handle to read from.
		is_binary (bool, optional): If True, indicates binary format (not supported 
			by this function). Defaults to False.
	
	Returns:
		tuple: A tuple containing:
			- label (str or None): The label/identifier for the vector, or None if EOF/error.
			- vec (np.ndarray or None): NumPy array of float32 values, or None if EOF/error.
	
	Raises:
		ParseException: If the line has fewer than 2 parts or cannot be parsed.
		
	Example:
		>>> file = open('vectors.txt', 'r')
		>>> label, vec = readvec(file)
		>>> print(f"Label: {label}, Vector shape: {vec.shape}")
		Label: word1, Vector shape: (300,)
	"""
	if is_binary:
		# For binary formats, this function shouldn't be called
		return None, None
	
	line = file.readline()
	if not line:
		return None, None

	split_char = config.params['split_char']

	if split_char == '' or split_char is None:# default: split on any whitespace
		parts = line.strip().split()
	else:
		parts = line.strip().split(split_char)
	if len(parts) < 2:
		print(line)
		print('no of parts=', len(parts))
		raise ParseException(parts)

	try:
		label = parts[0]
		vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
		return label, vec
	except ValueError:
		return None, None  # Skip lines with bad floats
	except ValueError:
		raise ParseException(parts)

class Reader:
	"""
	A versatile reader for loading vector data from multiple file formats.
	
	This class provides a unified interface for reading vector data and labels from various
	file formats including text files, CSV files, gzipped files, and NumPy binary files.
	The reader supports streaming data in batches to handle large datasets efficiently.

	Supported File Formats:
		- Text files (.txt): Space/tab delimited format
		- CSV files (.csv): Comma-separated values
		- Gzipped files (.txt.gz, .csv.gz): Compressed versions of above formats
		- NumPy files (.npy): Binary arrays with shape (n_samples, n_features+1)

	File Format Examples:
		Text/CSV format:
			word1 0.1 0.2 0.3 ... # optional comment
			word2 0.4 0.5 0.6 ...
			
		NumPy format:
			Array shape: (n_samples, n_features+1)
			First column: labels (converted to strings)
			Remaining columns: feature vectors

	Attributes:
		file_path (str): Path to the input file.
		file (TextIO or None): Open file handle (None for .npy files).
		counter (int): Total number of vectors successfully read.
		file_type (str): Type of file being read ('text', 'gzip', or 'npy').
		npy_data (np.ndarray or None): Loaded NumPy data for .npy files.
		npy_index (int): Current reading position in .npy data.
		
	Example:
		>>> reader = Reader('data.txt')
		>>> for vectors, labels in reader.stream_batches(100):
		...     print(f"Batch shape: {vectors.shape}, Labels: {len(labels)}")
		>>> reader.close()
	"""
	def __init__(self, file_path):
		"""
		Initialize the Reader with automatic file type detection.

		Opens the specified file and determines the appropriate reading strategy
		based on the file extension. Supports automatic decompression for gzipped files.

		Args:
			file_path (str): Absolute or relative path to the input file.
						   Supported extensions: .txt, .csv, .npy, .gz
						   
		Raises:
			FileNotFoundError: If the specified file does not exist.
			IOError: If the file cannot be opened or read.
			
		Example:
			>>> reader = Reader('/path/to/vectors.txt')
			>>> reader = Reader('/path/to/data.npy') 
			>>> reader = Reader('/path/to/compressed.txt.gz')
		"""
		self.file_path = file_path
		self.counter = 0
		self.npy_data = None
		self.npy_index = 0
		
		# Determine file type and open accordingly
		if file_path.endswith('.npy'):
			self.file_type = 'npy'
			self.npy_data = np.load(file_path)
			self.file = None
		elif file_path.endswith('.gz'):
			self.file_type = 'gzip'
			import gzip
			self.file = gzip.open(file_path, 'rt', encoding='utf-8')
		else:
			self.file_type = 'text'
			self.file = open(file_path, 'r', encoding='utf-8')

	def _read_npy_batch(self, batch_size):
		"""
		Read a batch of vectors from a NumPy (.npy) file.
		
		This internal method handles reading batches from pre-loaded NumPy arrays.
		It assumes the array has labels in the first column and features in remaining columns.
		
		Args:
			batch_size (int): Maximum number of vectors to read in this batch.
							Actual batch size may be smaller if near end of file.
			
		Returns:
			tuple: A tuple containing:
				- vectors (np.ndarray or None): Array of shape (batch_size, n_features) 
				  containing the feature vectors, or None if end of file reached.
				- labels (np.ndarray or None): Array of shape (batch_size,) containing 
				  string labels, or None if end of file reached.
				  
		Note:
			- If the .npy file contains only vectors (no labels), sequential indices 
			  are generated as string labels.
			- Handles both 1D and 2D NumPy arrays automatically.
		"""
		if self.npy_index >= len(self.npy_data):
			return None, None
			
		end_index = min(self.npy_index + batch_size, len(self.npy_data))
		batch_data = self.npy_data[self.npy_index:end_index]
		
		# Assume first column is labels, rest are features
		if batch_data.ndim == 2 and batch_data.shape[1] > 1:
			labels = batch_data[:, 0].astype(str)  # Convert to string labels
			vectors = batch_data[:, 1:].astype(np.float32)
		else:
			# If only one column or 1D array, treat as vectors with index as labels
			vectors = batch_data.astype(np.float32)
			if vectors.ndim == 1:
				vectors = vectors.reshape(-1, 1)
			labels = np.array([str(i) for i in range(self.npy_index, end_index)])
		
		self.npy_index = end_index
		self.counter += len(vectors)
		
		return vectors, labels

	def stream_batches(self, batch_size):
		"""
		Generate batches of vectors and labels from the input file.

		This generator method provides a memory-efficient way to process large datasets
		by reading and yielding data in configurable batch sizes. It automatically
		handles different file formats and provides progress feedback.

		Args:
			batch_size (int): Number of vectors to include in each batch.
							Must be a positive integer. The last batch may contain
							fewer vectors if the total number of vectors is not
							evenly divisible by batch_size.

		Yields:
			tuple: Each iteration yields a tuple containing:
				- vectors (np.ndarray): Array of shape (actual_batch_size, vector_dim) 
				  containing the feature vectors as float32 values.
				- labels (np.ndarray): Array of shape (actual_batch_size,) containing 
				  string labels corresponding to each vector.

		Raises:
			ParseException: If any line in text/CSV files cannot be parsed correctly.
			ValueError: If batch_size is not a positive integer.
			
		Example:
			>>> reader = Reader('embeddings.txt')
			>>> for vectors, labels in reader.stream_batches(1000):
			...     print(f"Processing batch: {vectors.shape}")
			...     # Process the batch of vectors and labels
			...     model.train_step(vectors, labels)
			
		Note:
			- Progress is automatically printed every config.params['batch_size'] vectors
			- For .npy files, batches are read directly from memory
			- For text/CSV/gzip files, batches are constructed by parsing individual lines
		"""
		if self.file_type == 'npy':
			while True:
				vectors, labels = self._read_npy_batch(batch_size)
				if vectors is None:
					break
				if self.counter % config.params['batch_size'] == 0:
					print(f"\rRead {self.counter} vectors", end='', flush=True)
				yield vectors, labels
		else:
			while True:
				vectors = []
				labels = []
				for _ in range(batch_size):
					label, vec = readvec(self.file, is_binary=(self.file_type == 'npy'))
					if vec is not None:
						vectors.append(vec)
						labels.append(label)
						self.counter += 1
						if self.counter % config.params['batch_size'] == 0:
							print(f"\rRead {self.counter} vectors", end='', flush=True)
				if not vectors:
					break
				yield np.stack(vectors), np.array(labels)

	def close(self):
		"""
		Close the file handle and release system resources.
		
		This method should be called when finished reading to properly close
		file handles and free system resources. It's safe to call multiple times.
		For .npy files (which don't have file handles), this method does nothing.
		
		Example:
			>>> reader = Reader('data.txt')
			>>> # ... process data ...
			>>> reader.close()  # Always close when done
			
		Note:
			Consider using the Reader in a context manager pattern:
			>>> with Reader('data.txt') as reader:  # If __enter__/__exit__ implemented
			...     for batch in reader.stream_batches(100):
			...         process(batch)
		"""
		if self.file is not None:
			self.file.close()

def set_of_points_from_file(filepath):
	reader = Reader(filepath)

	points = []
	for vectors, _ in reader.stream_batches(config.params['batch_size']):
		points.append(vectors)

	return setofpoints.SetOfPoints(np.vstack(points))


# ------------------- Main ---------------------
def main():
	"""
	Test function demonstrating the Reader class functionality.
	
	This function serves as both a test suite and usage example for the Reader class.
	It loads test configuration, initializes a Reader with sample data, and demonstrates
	basic batch processing functionality.
	
	The function tests various file formats and provides example usage patterns.
	Test file paths can be modified by uncommenting different options.
	
	Configuration:
		- Automatically sets split_char based on file extension
		- Uses test batch size of 100 vectors
		- Provides comprehensive error handling examples
		
	Example Output:
		Testing Reader...
		Read 100 vectors, shape = (100, 300)
		Sample labels: ['word1' 'word2' 'word3' 'word4' 'word5']
		Sample vector[0]: [0.1 0.2 0.3 ...]
		Test successful
	"""
	set_params()
	if config.params['test']:
		config.params['file_path'] = '../Voltage_Data/glove/glove_with_pos.txt'
		#config.params['file_path'] = '../Voltage_Data/mnist/mnist.csv'
		#config.params['file_path'] = '../Voltage_Data/data.npy'
		#config.params['file_path'] = '../Voltage_Data/data.txt.gz'
		config.params['batch_size'] = 100
		
		# Set split character based on file type
		if config.params['file_path'].endswith('.txt') or config.params['file_path'].endswith('.txt.gz'):
			config.params['split_char'] = ''
		elif config.params['file_path'].endswith('.csv') or config.params['file_path'].endswith('.csv.gz'):
			config.params['split_char'] = ','
		elif config.params['file_path'].endswith('.npy'):
			config.params['split_char'] = None  # Not used for .npy files
		else:
			config.params['split_char'] = ','  # Default to comma
		
		print("Testing Reader...")
		try:
			reader = Reader(config.params['file_path'])
			vectors, labels = next(reader.stream_batches(config.params['batch_size']))
			print(f"\nRead {len(vectors)} vectors, shape = {vectors.shape}")
			print("Sample labels:", labels[:5])
			print("Sample vector[0]:", vectors[0])
			reader.close()
			print("Test successful")
		except FileNotFoundError:
			print(f"File not found: {config.params['file_path']}")
		except ParseException as e:
			print(f"Parse error: {e}")
		except Exception as e:
			print(f"Unexpected error: {e}")

if __name__ == "__main__":
	main()


