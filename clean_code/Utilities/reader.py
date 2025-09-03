""" The Reader class for reading large text files in a memory-efficient manner.
"""

import numpy as np
import gzip
import os
import pandas as pd

import sys, os
sys.path.append(os.path.abspath("../clean_code/"))
from Utilities import config
from Utilities import set_params

class Reader:
    """
    Reader class for reading large text files in a memory-efficient manner.

    The main public methods are:
    - __init__(self, file_path): Initializes the Reader with the given file path.
    - stream_batches(self, batch_size): Generator that yields batches of vectors and additional fields.
    - peak_forward(self, n): Peeks ahead n lines without advancing the file pointer. Returns true if n rows remain.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        
        # Check if file is gzipped
        self.is_gzipped = file_path.endswith('.gz')
        
        # Pandas automatically handles gzipped files when compression is specified or inferred
        df = pd.read_csv(self.file_path, nrows=0)
        self._find_column_types()
        self.df = pd.read_csv(self.file_path, nrows=0, dtype=self.column_types)
        self.counter=0

        # Initialize persistent CSV reader
        self._csv_reader = None
        self._is_exhausted = False

    def _find_column_types(self):
        # Pandas automatically handles gzipped files
        self.df_sample = pd.read_csv(self.file_path, nrows=0,header=0)
        self.column_types = {col: 'float32' if col.startswith('d') else 'str' for col in self.df_sample.columns}
        # find range of columns that are float32, if this is not contiguous, raise an error
        self.float_cols = [i for i, col in enumerate(self.df_sample.columns) if col.startswith('d')]
        if self.float_cols != list(range(min(self.float_cols), max(self.float_cols)+1)):
            raise ValueError("Columns starting with 'd' must be contiguous")
        
        # Print file type for debugging
        file_type = "gzipped CSV" if self.is_gzipped else "CSV"
        print(f"Reading {file_type} file: {self.file_path}")
        return

    def stream_batches(self, batch_size):
        """ generate batches of vectors and additional fields from df.
        Use pandas read_csv as a generator with chunksize=batch_size.
        Maintains position between calls for persistent reading.
        """
        # Initialize the CSV reader if not already done
        if self._csv_reader is None and not self._is_exhausted:
            read_kwargs = {
                'chunksize': batch_size,
                'dtype': self.column_types,
                'on_bad_lines': 'skip',  # Skip rows that can't be parsed
                'engine': 'python'       # Required for on_bad_lines='skip'
            }
            self._csv_reader = pd.read_csv(self.file_path, **read_kwargs)
        

        # Continue reading from where we left off
        if self._csv_reader is not None and not self._is_exhausted:
            try:
                for chunk in self._csv_reader:
                    # extract vectors that are stored in columns starting with 'd'
                    vectors = chunk.iloc[:, self.float_cols].to_numpy()
                    # extract labels and other fields that are not part of the vectors
                    additional_fields = chunk.drop(columns=chunk.columns[self.float_cols]).to_dict(orient='records')
                    self.counter+=len(vectors)
                    yield vectors, additional_fields
            except StopIteration:
                self._is_exhausted = True
                self._csv_reader = None 
                
    
    def get_counter(self):
        """ Return the number of lines read so far. """
        return self.counter

    def peek_forward(self, n):
        """ Peek ahead n lines without advancing the file pointer.
        Returns true if n rows remain, false otherwise.
        """
        current_pos = self.df.index.stop
        self.df = pd.read_csv(self.file_path, skiprows=current_pos, nrows=n, dtype=self.column_types)
        has_n_rows = len(self.df) == n
        # reset df to original position
        self.df = pd.read_csv(self.file_path, skiprows=current_pos, nrows=0, dtype=self.column_types)
        return has_n_rows   
    
    def reset_reader(self):
        """Reset the CSV reader to start from the beginning of the file."""
        self._csv_reader = None
        self._is_exhausted = False
    
    def is_exhausted(self):
        """Check if the CSV reader has reached the end of the file."""
        return self._is_exhausted
    
    def close(self):
        """Close the CSV reader and clean up resources."""
        if self._csv_reader is not None:
            try:
                self._csv_reader.close()
            except AttributeError:
                pass  # Some pandas versions don't have close method
        self._csv_reader = None
        self._is_exhausted = False

class Writer:
    """ Writer class for writing batches of vectors and additional fields to a CSV file.
    """

    def __init__(self, output_path, reader: Reader):
        self.output_path = output_path
        self.reader = reader
        # If file exists, remove it to start fresh
        if os.path.exists(output_path):
            os.remove(output_path)

        # Track if header has been written
        self.header_written = False

    def write_header(self):
        """ Write the header to the output CSV file. The header is derived from the reader's dataframe, with the other fields coming first and the vector fields following
        This is called automatically by write_batch on the first call.
        """
        if self.header_written:
            return
        #
        header = list(self.reader.df_sample.columns)
        #reorganize the header so that the first columns are the non-vector columns, followed by the vector columns
        non_vector_cols = [col for col in header if not col.startswith('d')]
        vector_cols = [col for col in header if col.startswith('d')]
        self.header = non_vector_cols + vector_cols

        with open(self.output_path, 'w') as f:
            f.write(','.join(self.header) + '\n')
        self.header_written = True

    def write_row(self, vector, other):
        """ Write a single row to the writer's output file"""
        with open(self.output_path, 'a') as f:
            #generate a comma separated string from vector and other, with fields ordered as in self.header
            row = [other.get(col, '') for col in self.header if col in other] + list(vector)
            f.write(','.join(map(str, row)) + '\n')



def main():
    """
    Main function to demonstrate the usage of the Reader class.
    """
    set_params.set_params()
    
    # Test with regular CSV file
    #reader = Reader("../../Voltage_Data/glove/glove_with_pos_label.csv")
    reader = Reader("../../Voltage_Data/mnist/mnist.csv")
    
    # Uncomment to test with gzipped CSV file:
    # reader = Reader("../../Voltage_Data/glove/glove_with_pos_label.csv.gz")
    
    i=0
    batch_size=10
    from Utilities.timer import Timer
    timer = Timer()
    timer.mark("Reader initialized")

    for vectors, additional_fields in reader.stream_batches(batch_size=batch_size):
        print(f"i={i}, vectors.shape=",vectors.shape)
        print(f"additional_fields=",additional_fields)
        i+=1
        if i>5:
            break
    timer.mark(f"Read batch {i} of size {batch_size}")
#        if not reader.peek_forward(10*batch_size):  #using peek_forward is very time consuming. 
#                                                     better predefine number of rows for each phase
#            print(f"There are less than {10*batch_size} more rows to read.")
#            break

def batch_csv_example():
    """
    Example of how to use Reader and Writer for batch processing large CSV files.
    """
    # Initialize reader and writer
    reader = Reader("input.csv")
    writer = Writer("output.csv", reader)
    
    try:
        # Process file in batches
        for vectors, additional_fields in reader.stream_batches(batch_size=1000):
            # Process your data here (e.g., apply transformations)
            processed_vectors = vectors * 2  # Example transformation
            
            # Write batch to output file
            writer.write_batch(processed_vectors, additional_fields)
            
            print(f"Processed batch with {len(vectors)} rows")
    
    finally:
        # Clean up resources
        writer.close()
        reader.close()
        print("Batch processing completed")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        print(f"Empty data error: {e}")
    except pd.errors.ParserError as e:
        print(f"Parse error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        #write a traceback to stderr
        import traceback
        traceback.print_exc()