import numpy as np
import config
from set_params import set_params

class ParseException(Exception):
    pass

def readvec(file):
    line = file.readline()
    if not line:
        return None, None

    line = line.split('#')[0]
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
    Reads a text/csv file containing vectors line-by-line and yields batches of vectors and labels.

    Each line in the file should be in the format:
        word val1 val2 val3 ...
        pos val1 val2 val3 ... # originalword

    Attributes:
        file (TextIO): Opened file handle.
        counter (int): Number of vectors successfully read.
    """
    def __init__(self, file_path):
        """
        Initializes the Reader.

        Args:
            file_path (str): Path to the input text file.
        """
        self.file = open(file_path, 'r', encoding='utf-8')
        self.counter = 0

    def stream_batches(self, batch_size):
        """
        Generator that yields batches of vectors and labels as NumPy arrays.

        Args:
            batch_size (int): Number of vectors to include in each batch.

        Yields:
            tuple: (np.ndarray of shape (batch_size, vector_dim), np.ndarray of shape (batch_size,))
        """
        while True:
            vectors = []
            labels = []
            for _ in range(batch_size):
                label, vec = readvec(self.file)
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
        Closes the file handle.
        """
        self.file.close()

# ------------------- Main ---------------------
def main():
    set_params()
    if config.params['test']:
        config.params['file_path'] = '../Voltage_Data/glove/glove_with_pos.txt'
        #config.params['file_path'] = '../Voltage_Data/mnist/mnist.csv'
        config.params['batch_size'] = 100
        
        if config.params['file_path'].endswith('.txt'):
            config.params['split_char'] = ''
        else:
            config.params['split_char'] = ','
        
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


