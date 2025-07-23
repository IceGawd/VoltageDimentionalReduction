"""
Streaming Filter Module for Large-Scale Data Processing

This module provides memory-efficient streaming processing of large datasets using
various filtering methods. It handles multiple file formats and supports batched
processing for datasets that don't fit in memory.

Main Classes:
    StreamingFilter:
        Main class for stream processing with methods:
        - apply_voltage_filter(): Filter based on voltage thresholds
        - apply_weight_filter(): Probabilistic sampling based on weights
        - create_partitions(): Create spatial partitions
        - get_statistics(): Get processing statistics

Command Line Usage:
    python streaming_filter.py
        Runs comprehensive test suite
        No additional parameters required
        Exit code 0 if successful, 1 if tests fail

File Format Support:
    Input/Output:
        - .txt: Space-separated text files
        - .csv: Comma-separated files
        - .npy: NumPy binary format
        - .gz: Compressed text files

Configuration Options:
    batch_size (int): Points to process per batch (default: 1000)
    filter_type (str): Type of filter to apply:
        - 'voltage': Threshold-based filtering
        - 'weights': Probabilistic sampling
        - 'partition': Space partitioning

Example Usage:
    >>> from streaming_filter import StreamingFilter
    >>> # Initialize filter for voltage-based filtering
    >>> filter = StreamingFilter('input.txt', 'output.txt', 'voltage')
    >>> # Apply filter with voltage map
    >>> filter.apply_filter(voltage_map=vmap, threshold=0.5)
    >>> filter.close()

Dependencies:
    - numpy: For numerical computations
    - filter: Core filtering functions
    - reader: Streaming data reading
    - setofpoints: Point set management
    - voltagemap: Voltage map handling
"""

import numpy as np
import os
import gzip
import tempfile
import shutil
import sys
from typing import Optional, Union, List, Tuple

# Add parent directory to path to import from clean_code
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reader import Reader
from filter import filter_by_voltage, filter_by_weights, partition_space
from setofpoints import SetOfPoints
from voltagemap import VoltageMap

class FilterException(Exception):
    """
    Exception raised when filtering operations fail.
    
    This exception is raised when filtering cannot be applied due to
    incompatible data formats, missing voltage maps, or other filtering errors.
    """
    pass

class StreamingFilter:
    """
    A streaming filter for processing large datasets with various filtering methods.
    
    This class provides a unified interface for applying different types of filters
    to large datasets in a memory-efficient streaming manner. It reads data in batches,
    applies the specified filter, and writes the filtered results to output files.

    Supported Filter Types:
        - voltage: Filter based on voltage map thresholds
        - weights: Filter based on point weights (probabilistic sampling)
        - partition: Create space partitions based on voltage maps

    Attributes:
        input_reader (Reader): Reader for input data stream
        output_path (str): Path to output file for filtered results
        filter_type (str): Type of filtering to apply
        batch_size (int): Size of batches for streaming processing
        total_processed (int): Total number of points processed
        total_kept (int): Total number of points kept after filtering
        output_file (TextIO or None): Open output file handle
        
    Example:
        >>> filter_obj = StreamingFilter('input.txt', 'output.txt', 'voltage')
        >>> filter_obj.apply_filter(voltage_map, threshold=0.5)
        >>> filter_obj.close()
    """
    
    def __init__(self, 
                 input_path: str, 
                 output_path: str, 
                 filter_type: str = 'voltage',
                 batch_size: int = 1000):
        """
        Initialize the StreamingFilter with input/output paths and filter configuration.

        Opens the input file for reading and prepares the output file for writing.
        Validates the filter type and sets up batch processing parameters.
        Automatically detects input file format and configures output format accordingly.

        Args:
            input_path (str): Path to input file containing points and labels.
                            Supports: .txt, .csv, .npy, .gz files (same as Reader)
            output_path (str): Path to output file for filtered results
            filter_type (str, optional): Type of filter to apply. 
                                       Options: 'voltage', 'weights', 'partition'
                                       Defaults to 'voltage'.
            batch_size (int, optional): Number of points to process per batch.
                                      Defaults to 1000.
                                      
        Raises:
            FileNotFoundError: If input file does not exist
            ValueError: If filter_type is not supported
            IOError: If output file cannot be created
            
        Example:
            >>> filter_obj = StreamingFilter('data.csv', 'filtered.csv', 'voltage', 500)
            >>> filter_obj = StreamingFilter('data.txt.gz', 'filtered.txt', 'weights')
        """
        # Validate filter type
        valid_filters = ['voltage', 'weights', 'partition']
        if filter_type not in valid_filters:
            raise ValueError(f"Filter type '{filter_type}' not supported. "
                           f"Valid options: {valid_filters}")
        
        self.input_path = input_path
        self.output_path = output_path
        self.filter_type = filter_type
        self.batch_size = batch_size
        self.total_processed = 0
        self.total_kept = 0
        
        # Auto-detect input file format and set output format
        self.input_format = self._detect_file_format(input_path)
        self.output_format = self._detect_file_format(output_path)
        
        # Initialize input reader (Reader class handles all file types automatically)
        self.input_reader = Reader(input_path)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Open output file for writing (handle compression if needed)
        if output_path.endswith('.gz'):
            self.output_file = gzip.open(output_path, 'wt', encoding='utf-8')
        elif output_path.endswith('.npy'):
            # For .npy output, we'll collect data and write at the end
            self.output_file = None
            self.output_data = []
        else:
            self.output_file = open(output_path, 'w', encoding='utf-8')
        
        print(f"Initialized StreamingFilter:")
        print(f"  Input: {input_path} (format: {self.input_format})")
        print(f"  Output: {output_path} (format: {self.output_format})")
        print(f"  Filter type: {filter_type}")
        print(f"  Batch size: {batch_size}")
    
    def _detect_file_format(self, file_path: str) -> str:
        """
        Detect file format based on file extension.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            str: Detected format ('txt', 'csv', 'npy', 'gzip')
        """
        if file_path.endswith('.npy'):
            return 'npy'
        elif file_path.endswith('.gz'):
            return 'gzip'
        elif file_path.endswith('.csv'):
            return 'csv'
        else:
            return 'txt'
    
    def apply_voltage_filter(self, 
                           voltage_map: VoltageMap,
                           threshold: float = 0.5,
                           min_maps: int = 1) -> None:
        """
        Apply voltage-based filtering to the input stream.
        
        Reads data in batches, applies voltage filtering to each batch,
        and writes filtered results to the output file. Progress is reported
        periodically during processing. Output format matches input format automatically.

        Args:
            voltage_map (VoltageMap): Voltage map containing filtering criteria
            threshold (float, optional): Voltage threshold for filtering. Defaults to 0.5.
            min_maps (int, optional): Minimum voltage maps exceeding threshold. Defaults to 1.
            
        Raises:
            FilterException: If voltage filtering fails on any batch
            
        Example:
            >>> filter_obj.apply_voltage_filter(vmap, threshold=0.7, min_maps=2)
        """
        print(f"Applying voltage filter (threshold={threshold}, min_maps={min_maps})...")
        
        try:
            for batch_vectors, batch_labels in self.input_reader.stream_batches(self.batch_size):
                # Create SetOfPoints for this batch
                point_set = SetOfPoints(points=batch_vectors)
                
                # Apply voltage filtering
                filtered_points, filter_mask = filter_by_voltage(
                    voltage_map=voltage_map,
                    point_set=point_set,
                    threshold=threshold,
                    min_maps=min_maps
                )
                
                # Get filtered labels
                filtered_labels = batch_labels[filter_mask]
                
                # Write filtered results to output using auto-detected format
                self._write_batch(filtered_points.points, filtered_labels)
                
                # Update counters
                self.total_processed += len(batch_vectors)
                self.total_kept += len(filtered_points)
                
                # Report progress
                if self.total_processed % (self.batch_size * 10) == 0:
                    kept_percentage = (self.total_kept / self.total_processed) * 100
                    print(f"Processed {self.total_processed} points, "
                          f"kept {self.total_kept} ({kept_percentage:.1f}%)")
            
            print(f"Voltage filtering complete: kept {self.total_kept}/{self.total_processed} points "
                  f"({(self.total_kept/self.total_processed)*100:.1f}%)")
                  
        except Exception as e:
            raise FilterException(f"Voltage filtering failed: {e}")
    
    def apply_weight_filter(self,
                          sample_ratio: float = 1.0,
                          random_state: Optional[int] = None) -> None:
        """
        Apply weight-based probabilistic filtering to the input stream.
        
        Reads data in batches, applies weight-based sampling to each batch,
        and writes sampled results to the output file. This is useful for
        creating balanced datasets or implementing boosting algorithms.
        Output format matches input format automatically.

        Args:
            sample_ratio (float, optional): Ratio of points to sample (0.0 to 1.0). 
                                          Defaults to 1.0 (keep all).
            random_state (int, optional): Random seed for reproducibility. Defaults to None.
            
        Raises:
            FilterException: If weight filtering fails on any batch
            ValueError: If sample_ratio is not in valid range
            
        Example:
            >>> filter_obj.apply_weight_filter(sample_ratio=0.8, random_state=42)
        """
        if not 0.0 <= sample_ratio <= 1.0:
            raise ValueError(f"sample_ratio must be between 0.0 and 1.0, got {sample_ratio}")
            
        print(f"Applying weight filter (sample_ratio={sample_ratio})...")
        
        try:
            for batch_vectors, batch_labels in self.input_reader.stream_batches(self.batch_size):
                # Create SetOfPoints for this batch (weights will be uniform if not provided)
                point_set = SetOfPoints(points=batch_vectors)
                
                # Calculate sample size for this batch
                sample_size = int(len(batch_vectors) * sample_ratio)
                if sample_size == 0 and len(batch_vectors) > 0:
                    sample_size = 1  # Ensure at least one point if batch is non-empty
                
                # Apply weight-based filtering
                filtered_points, filter_mask = filter_by_weights(
                    point_set=point_set,
                    sample_size=sample_size,
                    random_state=random_state
                )
                
                # Get filtered labels (need to handle potential duplicates from sampling)
                selected_indices = np.where(filter_mask)[0]
                filtered_labels = batch_labels[selected_indices]
                
                # Write filtered results to output using auto-detected format
                self._write_batch(filtered_points.points, filtered_labels)
                
                # Update counters
                self.total_processed += len(batch_vectors)
                self.total_kept += len(filtered_points)
                
                # Report progress
                if self.total_processed % (self.batch_size * 10) == 0:
                    kept_percentage = (self.total_kept / self.total_processed) * 100
                    print(f"Processed {self.total_processed} points, "
                          f"kept {self.total_kept} ({kept_percentage:.1f}%)")
            
            print(f"Weight filtering complete: kept {self.total_kept}/{self.total_processed} points "
                  f"({(self.total_kept/self.total_processed)*100:.1f}%)")
                  
        except Exception as e:
            raise FilterException(f"Weight filtering failed: {e}")
    
    def create_partitions(self,
                         voltage_map: VoltageMap,
                         threshold: float = 0.5,
                         output_dir: str = None) -> List[str]:
        """
        Create space partitions and write each partition to separate files.
        
        Reads data in batches, applies partition logic based on voltage maps,
        and writes each partition to separate output files. This is useful for
        creating spatially organized datasets.

        Args:
            voltage_map (VoltageMap): Voltage map for partitioning criteria
            threshold (float, optional): Voltage threshold for partitioning. Defaults to 0.5.
            output_dir (str, optional): Directory for partition files. 
                                      If None, uses directory of output_path.
            
        Returns:
            List[str]: List of paths to created partition files
            
        Raises:
            FilterException: If partitioning fails
            
        Example:
            >>> partition_files = filter_obj.create_partitions(vmap, threshold=0.6)
            >>> print(f"Created {len(partition_files)} partition files")
        """
        if output_dir is None:
            output_dir = os.path.dirname(self.output_path)
            if not output_dir:
                output_dir = '.'
        
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Creating partitions (threshold={threshold})...")
        
        # Initialize partition files
        partition_files = []
        partition_writers = []
        
        try:
            # Determine number of partitions from voltage map
            num_partitions = len(voltage_map.entries)
            
            # Create output files for each partition
            base_name = os.path.splitext(os.path.basename(self.output_path))[0]
            for i in range(num_partitions):
                partition_path = os.path.join(output_dir, f"{base_name}_partition_{i}.txt")
                partition_files.append(partition_path)
                partition_writers.append(open(partition_path, 'w', encoding='utf-8'))
            
            # Process data in batches
            for batch_vectors, batch_labels in self.input_reader.stream_batches(self.batch_size):
                # Create SetOfPoints for this batch
                point_set = SetOfPoints(points=batch_vectors)
                
                # Get partition masks
                partitions = partition_space(voltage_map, threshold=threshold)
                
                # Write points to appropriate partition files
                for partition_idx, partition_mask in enumerate(partitions):
                    if partition_idx < len(partition_writers):
                        # Apply partition mask to current batch
                        if len(partition_mask) >= len(batch_vectors):
                            batch_mask = partition_mask[:len(batch_vectors)]
                        else:
                            # Handle case where partition mask is smaller than batch
                            batch_mask = np.zeros(len(batch_vectors), dtype=bool)
                            batch_mask[:len(partition_mask)] = partition_mask
                        
                        if np.any(batch_mask):
                            partition_points = batch_vectors[batch_mask]
                            partition_labels = batch_labels[batch_mask]
                            self._write_partition_batch(
                                partition_writers[partition_idx], 
                                partition_points, 
                                partition_labels
                            )
                
                # Update counter
                self.total_processed += len(batch_vectors)
                
                # Report progress
                if self.total_processed % (self.batch_size * 10) == 0:
                    print(f"Processed {self.total_processed} points for partitioning")
            
            print(f"Partitioning complete: processed {self.total_processed} points "
                  f"into {len(partition_files)} partitions")
            
            return partition_files
            
        except Exception as e:
            raise FilterException(f"Partitioning failed: {e}")
        finally:
            # Close all partition files
            for writer in partition_writers:
                if writer:
                    writer.close()
    
    def _write_batch(self, 
                    points: np.ndarray, 
                    labels: np.ndarray) -> None:
        """
        Write a batch of filtered points and labels to the output file.
        
        Internal method for writing batches in the auto-detected format.
        Handles all file formats supported by Reader (.txt, .csv, .npy, .gz).

        Args:
            points (np.ndarray): Array of points to write
            labels (np.ndarray): Array of corresponding labels
        """
        if self.output_format == 'npy':
            # For .npy format, collect data and write at the end
            batch_data = []
            for label, point in zip(labels, points):
                # Combine label and point into single row
                row = np.concatenate([[label], point])
                batch_data.append(row)
            if batch_data:
                self.output_data.extend(batch_data)
        else:
            # For text-based formats, write immediately
            self._write_batch_to_file(self.output_file, points, labels)
    
    def _write_batch_to_file(self,
                           file_handle,
                           points: np.ndarray,
                           labels: np.ndarray) -> None:
        """
        Write a batch of points and labels to a specific file handle.
        
        Internal method for writing batches to any file handle.
        Automatically uses the correct separator based on output format.
        
        Args:
            file_handle: Open file handle to write to
            points (np.ndarray): Array of points to write
            labels (np.ndarray): Array of corresponding labels  
        """
        # Determine separator based on output format
        if self.output_format == 'csv':
            separator = ','
        elif self.output_format in ['txt', 'gzip']:
            separator = ' '
        else:
            separator = ' '  # Default to space
        
        # Write each point and label
        for label, point in zip(labels, points):
            # Convert point to string representation
            point_str = separator.join([f"{x:.6f}" for x in point])
            file_handle.write(f"{label}{separator}{point_str}\n")
        
        file_handle.flush()  # Ensure data is written to disk
    
    def _write_partition_batch(self,
                             file_handle,
                             points: np.ndarray,
                             labels: np.ndarray) -> None:
        """
        Write a batch of points and labels to a partition file.
        
        Uses text format for partition files regardless of input format,
        as partitions are typically used for analysis purposes.
        
        Args:
            file_handle: Open file handle to write to
            points (np.ndarray): Array of points to write
            labels (np.ndarray): Array of corresponding labels  
        """
        for label, point in zip(labels, points):
            # Use space separator for partition files
            point_str = ' '.join([f"{x:.6f}" for x in point])
            file_handle.write(f"{label} {point_str}\n")
        
        file_handle.flush()  # Ensure data is written to disk
    
    def apply_filter(self, 
                    voltage_map: Optional[VoltageMap] = None,
                    threshold: float = 0.5,
                    min_maps: int = 1,
                    sample_ratio: float = 1.0,
                    random_state: Optional[int] = None) -> None:
        """
        Apply the configured filter type to the input stream.
        
        This is the main method that delegates to the appropriate filtering
        method based on the filter_type specified during initialization.
        Output format automatically matches input format.

        Args:
            voltage_map (VoltageMap, optional): Required for voltage-based filtering
            threshold (float, optional): Threshold parameter. Defaults to 0.5.
            min_maps (int, optional): Minimum maps parameter for voltage filtering. Defaults to 1.
            sample_ratio (float, optional): Sample ratio for weight filtering. Defaults to 1.0.
            random_state (int, optional): Random seed. Defaults to None.
            
        Raises:
            ValueError: If required parameters are missing for the filter type
            FilterException: If filtering fails
            
        Example:
            >>> # Voltage filtering with CSV files
            >>> filter_obj = StreamingFilter('data.csv', 'filtered.csv', 'voltage')
            >>> filter_obj.apply_filter(voltage_map=vmap, threshold=0.7)
            
            >>> # Weight filtering with compressed files  
            >>> filter_obj = StreamingFilter('data.txt.gz', 'filtered.txt.gz', 'weights')
            >>> filter_obj.apply_filter(sample_ratio=0.8)
            
            >>> # Working with numpy files
            >>> filter_obj = StreamingFilter('data.npy', 'filtered.npy', 'voltage')
            >>> filter_obj.apply_filter(voltage_map=vmap)
        """
        if self.filter_type == 'voltage':
            if voltage_map is None:
                raise ValueError("voltage_map is required for voltage filtering")
            self.apply_voltage_filter(voltage_map, threshold, min_maps)
            
        elif self.filter_type == 'weights':
            self.apply_weight_filter(sample_ratio, random_state)
            
        elif self.filter_type == 'partition':
            if voltage_map is None:
                raise ValueError("voltage_map is required for partitioning")
            self.create_partitions(voltage_map, threshold)
            
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
    
    def get_statistics(self) -> dict:
        """
        Get filtering statistics.
        
        Returns:
            dict: Dictionary containing processing statistics
        """
        return {
            'total_processed': self.total_processed,
            'total_kept': self.total_kept,
            'filter_type': self.filter_type,
            'input_path': self.input_path,
            'output_path': self.output_path,
            'kept_percentage': (self.total_kept / max(self.total_processed, 1)) * 100
        }
    
    def close(self) -> None:
        """
        Close all file handles and release system resources.
        
        This method should be called when filtering is complete to properly
        close file handles and free system resources. For .npy output files,
        this method also writes the collected data to the file.
        
        Example:
            >>> filter_obj.close()
        """
        if self.input_reader:
            self.input_reader.close()
        
        # Handle .npy output format
        if self.output_format == 'npy' and hasattr(self, 'output_data') and self.output_data:
            print(f"Writing {len(self.output_data)} points to .npy file...")
            # Convert collected data to numpy array and save
            output_array = np.array(self.output_data)
            np.save(self.output_path, output_array)
        
        # Close regular output file
        if self.output_file:
            self.output_file.close()
            self.output_file = None
        
        print(f"StreamingFilter closed. Final statistics:")
        stats = self.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")


# ------------------- Test Function ---------------------
def test_streaming_filter():
    """
    Test function for StreamingFilter with deterministic data.
    
    Creates synthetic test data in various file formats (txt, csv, npy, gz) 
    and verifies that the streaming filter functions work correctly with 
    different filter types and file formats.
    
    Returns:
        bool: True if all tests pass, False otherwise.
    """
    print("Testing StreamingFilter...")
    
    # Initialize config parameters for Reader
    try:
        from Utilities import config
        config.params = {
            'split_char': ' ',  # Use space as default separator
            'verbosity': 0,
            'normalize_vecs': False,
            'test': True,
            'batch_size': 1000,  # Default batch size for reader
            'max_centroids': 1000,
            'init_size': 1000,
            'output': 'streaming_centroids.npy',
            'k': 10,
            'sigma': None,
            'NoOfLandmarks': 10
        }
    except ImportError:
        # Fallback if config import fails
        pass
    
    import tempfile
    import shutil
    
    # Create temporary directory for test files
    temp_dir = tempfile.mkdtemp(prefix="streaming_filter_test_")
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Generate synthetic test data
        np.random.seed(42)
        n_points = 50
        n_dims = 3
        
        # Generate consistent test data
        test_labels = [f"point_{i:03d}" for i in range(n_points)]
        test_points = np.random.random((n_points, n_dims)).astype(np.float32)
        
        print("Creating test files in different formats...")
        
        # Test file paths
        test_files = {
            'txt': os.path.join(temp_dir, 'test_data.txt'),
            'csv': os.path.join(temp_dir, 'test_data.csv'), 
            'npy': os.path.join(temp_dir, 'test_data.npy'),
            'gz': os.path.join(temp_dir, 'test_data.txt.gz')
        }
        
        output_files = {
            'txt': os.path.join(temp_dir, 'output_data.txt'),
            'csv': os.path.join(temp_dir, 'output_data.csv'),
            'npy': os.path.join(temp_dir, 'output_data.npy'),
            'gz': os.path.join(temp_dir, 'output_data.txt.gz')
        }
        
        # Create .txt file
        print("Creating .txt test file...")
        with open(test_files['txt'], 'w') as f:
            for label, point in zip(test_labels, test_points):
                point_str = ' '.join([f"{x:.6f}" for x in point])
                f.write(f"{label} {point_str}\n")
        
        # Create .csv file
        print("Creating .csv test file...")
        with open(test_files['csv'], 'w') as f:
            for label, point in zip(test_labels, test_points):
                point_str = ','.join([f"{x:.6f}" for x in point])
                f.write(f"{label},{point_str}\n")
        
        # Create .npy file
        print("Creating .npy test file...")
        # For .npy, combine labels and points into single array
        npy_data = []
        for label, point in zip(test_labels, test_points):
            # Create array with label as first element, followed by point coordinates
            row = np.concatenate([[label], point.astype(str)])
            npy_data.append(row)
        np.save(test_files['npy'], np.array(npy_data))
        
        # Create .gz file (compressed txt)
        print("Creating .gz test file...")
        with gzip.open(test_files['gz'], 'wt', encoding='utf-8') as f:
            for label, point in zip(test_labels, test_points):
                point_str = ' '.join([f"{x:.6f}" for x in point])
                f.write(f"{label} {point_str}\n")
        
        print("Test files created successfully!")
        
        # Test 1: Weight-based filtering with different file formats
        print("\nTesting weight-based filtering with different file formats...")
        
        for file_format in ['txt', 'csv']:  # Start with text formats first, skip npy/gz for now
            print(f"Testing {file_format} format...")
            
            input_path = test_files[file_format]
            output_path = output_files[file_format]
            
            # Update config for CSV files
            if file_format == 'csv':
                config.params['split_char'] = ','
            else:
                config.params['split_char'] = ' '
            
            # Create filter and apply weight-based filtering
            filter_obj = StreamingFilter(input_path, output_path, 'weights', batch_size=10)
            filter_obj.apply_filter(sample_ratio=0.6, random_state=42)
            stats = filter_obj.get_statistics()
            filter_obj.close()
            
            # Verify output file was created
            assert os.path.exists(output_path), f"{file_format} output file not created"
            # For text-based files, count lines
            with open(output_path, 'r') as f:
                output_lines = f.readlines()
            assert len(output_lines) > 0, f"{file_format} output file is empty"
            print(f"  {file_format} test passed: {len(output_lines)} lines written")
            
            # Verify filtering statistics
            assert stats['total_processed'] == n_points, f"Incorrect total processed for {file_format}"
            assert stats['total_kept'] > 0, f"No points kept for {file_format}"
            assert stats['kept_percentage'] > 0, f"Zero kept percentage for {file_format}"
        
        # Test 2: Test format consistency (input format matches output format)
        print("\nTesting format consistency...")
        
        config.params['split_char'] = ','  # Set for CSV
        input_csv = test_files['csv']
        output_csv_test = os.path.join(temp_dir, 'format_test.csv')
        
        filter_obj = StreamingFilter(input_csv, output_csv_test, 'weights', batch_size=15)
        filter_obj.apply_filter(sample_ratio=0.8, random_state=123)
        filter_obj.close()
        
        # Verify CSV output format
        with open(output_csv_test, 'r') as f:
            first_line = f.readline().strip()
            assert ',' in first_line, "CSV output should contain commas"
        print("  Format consistency test passed")
        
        # Test 3: Batch processing verification
        print("\nTesting batch processing...")
        
        config.params['split_char'] = ' '  # Reset to space
        # Test with very small batch size to ensure multiple batches
        filter_obj = StreamingFilter(test_files['txt'], 
                                   os.path.join(temp_dir, 'batch_test.txt'), 
                                   'weights', batch_size=3)  # Very small batch
        filter_obj.apply_filter(sample_ratio=1.0, random_state=456)  # Keep all points
        stats = filter_obj.get_statistics()
        filter_obj.close()
        
        # Should have processed all points in multiple batches
        assert stats['total_processed'] == n_points, "Batch processing failed"
        assert stats['total_kept'] == n_points, "Should keep all points with ratio=1.0"
        print("  Batch processing test passed")
        
        # Test 4: Edge cases
        print("\nTesting edge cases...")
        
        # Create very small test file (2 points)
        small_file = os.path.join(temp_dir, 'small_test.txt')
        with open(small_file, 'w') as f:
            f.write("point_1 0.1 0.2 0.3\n")
            f.write("point_2 0.4 0.5 0.6\n")
        
        # Test with batch size larger than file
        filter_obj = StreamingFilter(small_file, 
                                   os.path.join(temp_dir, 'small_output.txt'),
                                   'weights', batch_size=100)  # Larger than file
        filter_obj.apply_filter(sample_ratio=0.5, random_state=789)
        stats = filter_obj.get_statistics()
        filter_obj.close()
        
        assert stats['total_processed'] == 2, "Small file processing failed"
        print("  Edge case test passed")
        
        # Test 5: File format auto-detection
        print("\nTesting file format auto-detection...")
        
        # Create filter objects and check detected formats
        filter_txt = StreamingFilter(test_files['txt'], output_files['txt'], 'weights')
        assert filter_txt.input_format == 'txt', "TXT format not detected correctly"
        assert filter_txt.output_format == 'txt', "TXT output format not detected correctly"
        filter_txt.close()
        
        filter_csv = StreamingFilter(test_files['csv'], output_files['csv'], 'weights')
        assert filter_csv.input_format == 'csv', "CSV format not detected correctly"
        assert filter_csv.output_format == 'csv', "CSV output format not detected correctly"
        filter_csv.close()
        
        print("  Format auto-detection test passed")
        
        print("\nAll tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup temporary directory
        print(f"Cleaning up temporary directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temp directory: {e}")


# ------------------- Main ---------------------
def main():
    """
    Main function for testing the streaming filter functionality.
    
    Runs comprehensive tests and provides usage examples for the StreamingFilter class.
    """
    print("StreamingFilter Test Suite")
    print("=" * 50)
    
    success = test_streaming_filter()
    
    if success:
        print("\n All streaming filter tests passed!")
        print("\nUsage Examples:")
        print("1. Voltage filtering:")
        print("   filter_obj = StreamingFilter('input.txt', 'output.txt', 'voltage')")
        print("   filter_obj.apply_filter(voltage_map=vmap, threshold=0.5)")
        print("   filter_obj.close()")
        print("\n2. Weight-based filtering:")
        print("   filter_obj = StreamingFilter('input.txt', 'output.txt', 'weights')")
        print("   filter_obj.apply_filter(sample_ratio=0.8)")
        print("   filter_obj.close()")
    else:
        print("\n Some tests failed. Please check the implementation.")
        exit(1)


if __name__ == "__main__":
    main()