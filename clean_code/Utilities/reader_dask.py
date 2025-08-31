import numpy as np
import pandas as pd
import dask.dataframe as dd
import config
from set_params import set_params

class ParseException(Exception):
    """
    Exception raised when parsing vector data from CSV files fails.
    
    This exception is raised when data cannot be parsed into the expected format.
    """
    pass

class DaskReader:
    """
    A high-performance reader for loading very large CSV files using Dask.
    
    This class provides an interface for reading vector data and labels from large CSV files
    using Dask for parallel processing and streaming pandas DataFrames. It efficiently handles
    datasets that don't fit in memory by processing them in parallel chunks.

    Supported File Formats:
        - CSV files (.csv): Comma-separated values
        - Gzipped CSV files (.csv.gz): Compressed CSV files

    File Format Examples:
        CSV format without header:
            word1,0.1,0.2,0.3
            word2,0.4,0.5,0.6
            
        CSV format with header:
            label,feature1,feature2,feature3
            word1,0.1,0.2,0.3
            word2,0.4,0.5,0.6
            
        CSV format with optional word column:
            word1,0.1,0.2,0.3,hello
            word2,0.4,0.5,0.6,world
            
        CSV format with header and word column:
            label,feature1,feature2,feature3,word
            word1,0.1,0.2,0.3,hello
            word2,0.4,0.5,0.6,world

    Attributes:
        file_path (str): Path to the input CSV file.
        dask_df (dask.dataframe.DataFrame): Dask DataFrame for parallel processing.
        counter (int): Total number of vectors successfully read.
        file_type (str): Type of file being read ('csv' or 'gzip').
        has_header (bool): Whether the CSV file has a header row.
        label_column (str): Name of the label column.
        feature_columns (list): List of feature column names.
        word_column (str or None): Name of the word column if present.
        has_word_column (bool): Whether the CSV file has a word column.
        
    Example:
        >>> reader = DaskReader('large_data.csv', has_header=True, chunk_size='100MB')
        >>> for df_chunk in reader.stream_pandas_chunks(1000):
        ...     vectors, labels, words = reader.dataframe_to_arrays(df_chunk)
        ...     print(f"Batch shape: {vectors.shape}, Labels: {len(labels)}, Words: {len(words) if words is not None else 'None'}")
        >>> reader.close()
    """
    
    def __init__(self, file_path, has_header=False, chunk_size='64MB', label_column=None):
        """
        Initialize the DaskReader with automatic file type detection and Dask setup.

        Args:
            file_path (str): Absolute or relative path to the input CSV file.
                           Supported extensions: .csv, .csv.gz
            has_header (bool, optional): Whether the CSV file has a header row.
                                       Defaults to False.
            chunk_size (str, optional): Size of each partition for Dask processing.
                                      Can use '64MB', '100MB', etc. Defaults to '64MB'.
            label_column (str, optional): Name of the label column if has_header=True.
                                        If None, assumes first column is label.
                           
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is not a supported CSV format.
            
        Example:
            >>> reader = DaskReader('/path/to/data.csv')
            >>> reader = DaskReader('/path/to/data_with_header.csv', has_header=True)
            >>> reader = DaskReader('/path/to/large_file.csv', chunk_size='200MB')
        """
        self.file_path = file_path
        self.counter = 0
        self.has_header = has_header
        self.label_column = label_column
        self.feature_columns = []
        self.word_column = None
        self.has_word_column = False
        
        # Check for supported file types
        if not (file_path.endswith('.csv') or file_path.endswith('.csv.gz')):
            raise ValueError(f"Unsupported file format. Only .csv and .csv.gz files are supported. Got: {file_path}")
        
        # Determine file type
        self.file_type = 'gzip' if file_path.endswith('.csv.gz') else 'csv'
        
        # Set up Dask DataFrame reading parameters
        read_kwargs = {
            'blocksize': chunk_size,
            'sep': config.params.get('split_char', ',') if config.params.get('split_char') != '' else ',',
        }
        
        # Handle header configuration
        if has_header:
            read_kwargs['header'] = 0
        else:
            read_kwargs['header'] = None
            
        try:
            # Create Dask DataFrame
            self.dask_df = dd.read_csv(file_path, **read_kwargs)
            self._setup_columns()
            print(f"Initialized Dask reader with {self.dask_df.npartitions} partitions")
            
        except Exception as e:
            raise ValueError(f"Failed to read CSV file with Dask: {e}")

    def _setup_columns(self):
        """
        Set up column names and identify label/feature/word columns.
        """
        columns = list(self.dask_df.columns)
        
        if self.has_header:
            # Use actual column names from header
            if self.label_column:
                if self.label_column not in columns:
                    raise ValueError(f"Label column '{self.label_column}' not found in file")
            else:
                self.label_column = columns[0]  # Assume first column is label
            
            # Check for word column (last column if it's named 'word' or contains string data)
            if 'word' in columns:
                self.word_column = 'word'
                self.has_word_column = True
            else:
                # Check if last column contains non-numeric data (potential word column)
                try:
                    last_col = columns[-1]
                    sample_data = self.dask_df[last_col].head(10)
                    # Try to convert to float, if it fails, it might be a word column
                    pd.to_numeric(sample_data, errors='raise')
                    # If successful, it's not a word column
                    self.has_word_column = False
                except (ValueError, TypeError):
                    # If conversion fails, treat as word column
                    self.word_column = columns[-1]
                    self.has_word_column = True
            
            # Feature columns are everything except label and word columns
            excluded_cols = [self.label_column]
            if self.has_word_column:
                excluded_cols.append(self.word_column)
            self.feature_columns = [col for col in columns if col not in excluded_cols]
            
        else:
            # Generate column names for headerless files
            self.label_column = columns[0]
            
            # Check if last column contains non-numeric data
            try:
                last_col = columns[-1]
                sample_data = self.dask_df[last_col].head(10)
                pd.to_numeric(sample_data, errors='raise')
                # If successful, no word column
                self.feature_columns = columns[1:]
                self.has_word_column = False
            except (ValueError, TypeError):
                # If conversion fails, last column is word column
                self.word_column = columns[-1]
                self.has_word_column = True
                self.feature_columns = columns[1:-1]  # Exclude both label and word columns
            
        print(f"Label column: {self.label_column}")
        print(f"Feature columns: {len(self.feature_columns)} columns")
        if self.has_word_column:
            print(f"Word column: {self.word_column}")
        else:
            print("No word column detected")

    def get_dask_dataframe(self):
        """
        Get the underlying Dask DataFrame for advanced operations.
        
        Returns:
            dask.dataframe.DataFrame: The Dask DataFrame for the CSV file.
            
        Example:
            >>> reader = DaskReader('data.csv')
            >>> df = reader.get_dask_dataframe()
            >>> result = df.groupby('label').mean().compute()
        """
        return self.dask_df

    def stream_pandas_chunks(self, rows_per_chunk=1000):
        """
        Stream the data as pandas DataFrame chunks for processing.

        This generator yields pandas DataFrames by computing Dask partitions
        and optionally splitting them into smaller chunks for memory efficiency.

        Args:
            rows_per_chunk (int): Target number of rows per pandas DataFrame chunk.
                                If None, yields entire partitions.

        Yields:
            pd.DataFrame: Pandas DataFrame chunks ready for processing.
            
        Example:
            >>> reader = DaskReader('large_file.csv')
            >>> for df_chunk in reader.stream_pandas_chunks(5000):
            ...     # Process pandas DataFrame chunk
            ...     print(f"Processing {len(df_chunk)} rows")
        """
        for i in range(self.dask_df.npartitions):
            # Compute one partition at a time to manage memory
            partition_df = self.dask_df.get_partition(i).compute()
            
            if rows_per_chunk is None or len(partition_df) <= rows_per_chunk:
                yield partition_df
                self.counter += len(partition_df)
            else:
                # Split large partitions into smaller chunks
                for start_idx in range(0, len(partition_df), rows_per_chunk):
                    end_idx = min(start_idx + rows_per_chunk, len(partition_df))
                    chunk = partition_df.iloc[start_idx:end_idx]
                    yield chunk
                    self.counter += len(chunk)
            
            # Print progress
            if self.counter % config.params.get('batch_size', 1000) == 0:
                print(f"\rProcessed {self.counter} rows", end='', flush=True)

    def dataframe_to_arrays(self, df_chunk):
        """
        Convert a pandas DataFrame chunk to numpy arrays (vectors, labels, and optionally words).
        
        Args:
            df_chunk (pd.DataFrame): Pandas DataFrame chunk to convert.
            
        Returns:
            tuple: A tuple containing:
                - vectors (np.ndarray): Array of shape (n_rows, n_features) with float32 values.
                - labels (np.ndarray): Array of shape (n_rows,) with string labels.
                - words (np.ndarray or None): Array of shape (n_rows,) with string words, 
                  or None if no word column exists.
                
        Example:
            >>> reader = DaskReader('data.csv', has_header=True)
            >>> for df_chunk in reader.stream_pandas_chunks(1000):
            ...     vectors, labels, words = reader.dataframe_to_arrays(df_chunk)
            ...     print(f"Vectors shape: {vectors.shape}")
            ...     if words is not None:
            ...         print(f"Words: {words[:5]}")
        """
        try:
            # Extract labels
            labels = df_chunk[self.label_column].astype(str).values
            
            # Extract feature vectors
            feature_data = df_chunk[self.feature_columns]
            vectors = feature_data.astype(np.float32).values
            
            # Extract words if word column exists
            words = None
            if self.has_word_column:
                words = df_chunk[self.word_column].astype(str).values
            
            return vectors, labels, words
            
        except Exception as e:
            raise ParseException(f"Failed to convert DataFrame to arrays: {e}")

    def stream_batches(self, batch_size):
        """
        Generate batches of vectors and labels (compatible with original Reader interface).

        This method maintains compatibility with the original Reader interface while
        using Dask for efficient parallel processing underneath. If word column exists,
        it returns a third element in the tuple.

        Args:
            batch_size (int): Number of vectors to include in each batch.

        Yields:
            tuple: Each iteration yields a tuple containing:
                - vectors (np.ndarray): Array of shape (batch_size, vector_dim) 
                  containing the feature vectors as float32 values.
                - labels (np.ndarray): Array of shape (batch_size,) containing 
                  string labels corresponding to each vector.
                - words (np.ndarray, optional): Array of shape (batch_size,) containing
                  string words, only included if has_word_column is True.

        Example:
            >>> reader = DaskReader('embeddings.csv')
            >>> for batch_data in reader.stream_batches(1000):
            ...     if len(batch_data) == 3:
            ...         vectors, labels, words = batch_data
            ...         print(f"Processing batch: {vectors.shape}, words: {len(words)}")
            ...     else:
            ...         vectors, labels = batch_data
            ...         print(f"Processing batch: {vectors.shape}")
        """
        current_vectors = []
        current_labels = []
        current_words = [] if self.has_word_column else None
        
        for df_chunk in self.stream_pandas_chunks(rows_per_chunk=batch_size * 2):
            vectors, labels, words = self.dataframe_to_arrays(df_chunk)
            
            # Add to current batch
            current_vectors.extend(vectors)
            current_labels.extend(labels)
            if self.has_word_column:
                current_words.extend(words)
            
            # Yield complete batches
            while len(current_vectors) >= batch_size:
                batch_vectors = np.array(current_vectors[:batch_size], dtype=np.float32)
                batch_labels = np.array(current_labels[:batch_size])
                
                current_vectors = current_vectors[batch_size:]
                current_labels = current_labels[batch_size:]
                
                if self.has_word_column:
                    batch_words = np.array(current_words[:batch_size])
                    current_words = current_words[batch_size:]
                    yield batch_vectors, batch_labels, batch_words
                else:
                    yield batch_vectors, batch_labels
        
        # Yield remaining data as final batch
        if current_vectors:
            batch_vectors = np.array(current_vectors, dtype=np.float32)
            batch_labels = np.array(current_labels)
            
            if self.has_word_column:
                batch_words = np.array(current_words)
                yield batch_vectors, batch_labels, batch_words
            else:
                yield batch_vectors, batch_labels

    def get_column_info(self):
        """
        Get information about the columns in the CSV file.
        
        Returns:
            dict: Dictionary containing column information with keys:
                - 'label_column': Name of the label column
                - 'feature_columns': List of feature column names  
                - 'word_column': Name of the word column (None if not present)
                - 'total_columns': Total number of columns
                - 'has_header': Whether file has header
                - 'has_word_column': Whether file has word column
        """
        return {
            'label_column': self.label_column,
            'feature_columns': self.feature_columns,
            'word_column': self.word_column,
            'total_columns': len(self.feature_columns) + 1 + (1 if self.has_word_column else 0),
            'has_header': self.has_header,
            'has_word_column': self.has_word_column
        }

    def get_file_info(self):
        """
        Get information about the file and Dask configuration.
        
        Returns:
            dict: Dictionary containing file information.
        """
        return {
            'file_path': self.file_path,
            'file_type': self.file_type,
            'partitions': self.dask_df.npartitions,
            'estimated_rows': len(self.dask_df) if hasattr(self.dask_df, '__len__') else 'Unknown'
        }

    def compute_statistics(self):
        """
        Compute basic statistics for the dataset using Dask's parallel processing.
        
        Returns:
            dict: Dictionary containing dataset statistics.
            
        Example:
            >>> reader = DaskReader('data.csv', has_header=True)
            >>> stats = reader.compute_statistics()
            >>> print(f"Dataset has {stats['total_rows']} rows")
        """
        try:
            # Compute basic statistics in parallel
            total_rows = len(self.dask_df)
            label_counts = self.dask_df[self.label_column].value_counts().compute()
            
            # Feature statistics
            feature_stats = self.dask_df[self.feature_columns].describe().compute()
            
            return {
                'total_rows': total_rows,
                'total_features': len(self.feature_columns),
                'label_counts': label_counts.to_dict(),
                'feature_statistics': feature_stats
            }
        except Exception as e:
            print(f"Warning: Could not compute statistics: {e}")
            return {}

    def get_counter(self):
        """
        Get the total number of vectors read so far.
        
        Returns:
            int: Total count of vectors successfully read from the file.
        """
        return self.counter
        
    def close(self):
        """
        Clean up resources.
        
        Note: Dask DataFrames don't require explicit closing, but this method
        is provided for compatibility with the original Reader interface.
        """
        print(f"\nDask reader processed {self.counter} total rows")
        # Dask DataFrames are automatically garbage collected

    def extract_batch_components(self, batch_data):
        """
        Helper method to extract components from batch data for backward compatibility.
        
        Args:
            batch_data (tuple): Batch data returned from stream_batches()
            
        Returns:
            tuple: Always returns (vectors, labels, words) where words may be None
            
        Example:
            >>> reader = DaskReader('data.csv')
            >>> batch_data = next(reader.stream_batches(100))
            >>> vectors, labels, words = reader.extract_batch_components(batch_data)
        """
        if len(batch_data) == 3:
            return batch_data  # vectors, labels, words
        else:
            vectors, labels = batch_data
            return vectors, labels, None

# For backward compatibility, create an alias
Reader = DaskReader

# ------------------- Main ---------------------
def main():
    """
    Test function demonstrating the DaskReader class functionality.
    
    This function serves as both a test suite and usage example for the DaskReader class.
    It loads test configuration, initializes a DaskReader with sample CSV data, and demonstrates
    parallel processing capabilities with both pandas DataFrame chunks and numpy array batches.
    
    The function tests various features including:
    - Basic batch processing (compatible with original Reader)
    - Pandas DataFrame streaming
    - Column information extraction
    - Dataset statistics computation
    - Performance comparison
    
    Configuration:
        - Automatically sets split_char to ',' for CSV files
        - Uses test batch size from config
        - Demonstrates both header and headerless file handling
        
    Example Output:
        Testing DaskReader...
        Initialized Dask reader with 8 partitions
        Column info: {'label_column': '0', 'feature_columns': [...], ...}
        Read 100 vectors, shape = (100, 784)
        Processing pandas chunks...
        Dataset statistics: {'total_rows': 1000, ...}
        Test successful
    """
    set_params()
    if config.params['test']:
        config.params['file_path'] = '../Voltage_Data/mnist/mnist.csv'
        #config.params['file_path'] = '../Voltage_Data/data.csv.gz'
        config.params['batch_size'] = 100
        
        # Set split character for CSV files
        config.params['split_char'] = ','
        
        print("Testing DaskReader...")
        try:
            # Test 1: Basic usage (compatible with original Reader interface)
            print("\n=== Test 1: Basic batch processing ===")
            reader = DaskReader(config.params['file_path'], chunk_size='32MB')
            
            # Get file and column information
            file_info = reader.get_file_info()
            column_info = reader.get_column_info()
            print(f"File info: {file_info}")
            print(f"Column info: {column_info}")
            
            # Test batch processing (original interface)
            batch_data = next(reader.stream_batches(config.params['batch_size']))
            if len(batch_data) == 3:
                vectors, labels, words = batch_data
                print(f"Read {len(vectors)} vectors, shape = {vectors.shape}")
                print("Sample labels:", labels[:5])
                print("Sample words:", words[:5])
                print("Sample vector[0] (first 5 features):", vectors[0][:5])
            else:
                vectors, labels = batch_data
                print(f"Read {len(vectors)} vectors, shape = {vectors.shape}")
                print("Sample labels:", labels[:5])
                print("Sample vector[0] (first 5 features):", vectors[0][:5])
            
            # Test 2: Pandas DataFrame streaming
            print("\n=== Test 2: Pandas DataFrame streaming ===")
            chunk_count = 0
            total_rows = 0
            
            for df_chunk in reader.stream_pandas_chunks(rows_per_chunk=200):
                vectors, labels, words = reader.dataframe_to_arrays(df_chunk)
                total_rows += len(df_chunk)
                chunk_count += 1
                word_info = f", words: {words[:3] if words is not None else 'None'}"
                print(f"Chunk {chunk_count}: {len(df_chunk)} rows, vectors shape: {vectors.shape}{word_info}")
                
                if chunk_count >= 3:  # Limit output for testing
                    break
            
            print(f"Processed {total_rows} rows in {chunk_count} chunks")
            
            # Test 3: Dataset statistics (if not too large)
            print("\n=== Test 3: Dataset statistics ===")
            try:
                stats = reader.compute_statistics()
                if stats:
                    print(f"Total rows: {stats.get('total_rows', 'Unknown')}")
                    print(f"Total features: {stats.get('total_features', 'Unknown')}")
                    print(f"Unique labels: {len(stats.get('label_counts', {}))}")
            except Exception as e:
                print(f"Skipping statistics computation: {e}")
            
            # Test 4: Test with header (uncomment if CSV has headers)
            # print("\n=== Test 4: CSV with headers ===")
            # reader_with_header = DaskReader(config.params['file_path'], 
            #                               has_header=True, 
            #                               label_column='label')
            # column_info = reader_with_header.get_column_info()
            # print(f"Header-based column info: {column_info}")
            # reader_with_header.close()
            
            reader.close()
            print("\n=== All tests successful! ===")
            
        except FileNotFoundError:
            print(f"File not found: {config.params['file_path']}")
            print("Note: Make sure the test CSV file exists or update the file path")
        except ImportError as e:
            print(f"Import error: {e}")
            print("Please install Dask: pip install dask[dataframe]")
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()


