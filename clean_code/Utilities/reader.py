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

    def _find_column_types(self, sample_size=1000):
        # Pandas automatically handles gzipped files
        df_sample = pd.read_csv(self.file_path, nrows=0)
        self.column_types = {col: 'float32' if col.startswith('d') else 'str' for col in df_sample.columns}
        # find range of columns that are float32, if this is not contiguous, raise an error
        self.float_cols = [i for i, col in enumerate(df_sample.columns) if col.startswith('d')]
        if self.float_cols != list(range(min(self.float_cols), max(self.float_cols)+1)):
            raise ValueError("Columns starting with 'd' must be contiguous")
        
        # Print file type for debugging
        file_type = "gzipped CSV" if self.is_gzipped else "CSV"
        print(f"Reading {file_type} file: {self.file_path}")
        return

    def stream_batches(self, batch_size):
        """ generate batches of vectors and additional fields from df.
        Use pandas read_csv as a generator with chunksize=batch_size.
        """
        for chunk in pd.read_csv(self.file_path, chunksize=batch_size, dtype=self.column_types):
            # extract vectors that are stored in columns starting with 'd'
            vectors = chunk.iloc[:, self.float_cols].to_numpy()
            # extract labels and othere fields that are not part of the vectors
            additional_fields = chunk.drop(columns=chunk.columns[self.float_cols]).to_dict(orient='records')
            yield vectors, additional_fields

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

def main():
    """
    Main function to demonstrate the usage of the Reader class.
    """
    set_params.set_params()
    
    # Test with regular CSV file
    # reader = Reader("../../Voltage_Data/glove/glove_with_pos_label.csv")
    
    # Uncomment to test with gzipped CSV file:
    reader = Reader("../../Voltage_Data/glove/glove_with_pos_label.csv.gz")
    
    i=0
    batch_size=10000
    from Utilities.timer import Timer
    timer = Timer()
    timer.mark("Reader initialized")
    print(f"config.params['file_path']={config.params['file_path']}")
    # print(f"config.params.keys()={config.params.keys()}")
    for vectors, additional_fields in reader.stream_batches(batch_size=batch_size):
        print(f"i={i}, vectors.shape=",vectors.shape)
        #print(f"additional_fields=",additional_fields)
        i+=1
    timer.mark(f"Read batch {i} of size {batch_size}")
#        if not reader.peek_forward(10*batch_size):
#            print(f"There are less than {10*batch_size} more rows to read.")
#            break

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