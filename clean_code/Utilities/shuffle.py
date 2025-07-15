import os
import tempfile
import random
import heapq
import shutil
import argparse

def tag_lines_with_random_keys(input_path, temp_dir, lines_per_chunk=100000):
    """
    Reads a large text file in chunks, tags each line with a random key, and writes each sorted chunk to a temporary file.

    Args:
        input_path (str): Path to the input text file.
        temp_dir (str): Path to a temporary directory where chunk files will be stored.
        lines_per_chunk (int): Number of lines to process per chunk (default: 100000).

    Returns:
        List[str]: List of file paths to the sorted chunk files.
    """
    chunk_files = []
    with open(input_path, 'r') as infile:
        while True:
            lines = []
            try:
                for _ in range(lines_per_chunk):
                    line = next(infile)
                    key = random.random()
                    lines.append((key, line))
            except StopIteration:
                pass

            if not lines:
                break

            # Sort lines by random key for efficient merging
            lines.sort()
            temp_path = os.path.join(temp_dir, next(tempfile._get_candidate_names()) + ".txt")
            with open(temp_path, 'w') as f:
                for key, line in lines:
                    f.write(f"{key:.17f}\t{line}")
            chunk_files.append(temp_path)
    return chunk_files

def merge_sorted_chunks(chunk_files, output_path):
    """
    Merges sorted chunk files into a single shuffled output file using a heap-based external merge.

    Args:
        chunk_files (List[str]): List of sorted temporary file paths.
        output_path (str): Path to the output file where merged content will be written.
    """
    def line_iter(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                key_str, content = line.split('\t', 1)
                yield (float(key_str), content)

    with open(output_path, 'w') as outfile:
        for _, line in heapq.merge(*(line_iter(fp) for fp in chunk_files)):
            outfile.write(line)

def shuffle_large_file_external(input_path, output_path):
    """
    Shuffles a large file by tagging each line with a random key and using an external merge sort.

    Args:
        input_path (str): Path to the input file.
        output_path (str): Path to the output file for shuffled lines.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_files = tag_lines_with_random_keys(input_path, temp_dir)
        merge_sorted_chunks(chunk_files, output_path)

def main():
    """
    Main function to parse command-line arguments and shuffle a large input file.
    """
    parser = argparse.ArgumentParser(description="Shuffle a large file using external sorting.")
    parser.add_argument("input_path", help="Path to the input text file.")
    parser.add_argument("output_path", help="Path to the output shuffled file.")
    parser.add_argument("--lines-per-chunk", type=int, default=100000, help="Number of lines per chunk (default: 100000).")
    args = parser.parse_args()

    shuffle_large_file_external(args.input_path, args.output_path)

if __name__ == "__main__":
    main()
