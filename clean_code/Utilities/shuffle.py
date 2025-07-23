"""
External Memory File Shuffling Utility

This module provides functionality to efficiently shuffle large text files that don't fit in memory.
It uses an external merge sort approach by splitting the file into chunks, tagging lines with random
keys, and then merging them back together in a memory-efficient way.

Example:
    >>> from shuffle import shuffle_large_file_external
    >>> shuffle_large_file_external('input.txt', 'shuffled_output.txt')

The implementation uses a two-phase approach:
1. Split and tag: File is read in chunks, each line gets a random key
2. External merge: Chunks are merged using a heap-based approach
"""

import os
import tempfile
import random
import heapq
import shutil
import argparse

def tag_lines_with_random_keys(input_path: str, temp_dir: str, lines_per_chunk: int = 100000) -> list[str]:
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

def merge_sorted_chunks(chunk_files: list[str], output_path: str) -> None:
    """
    Merges sorted chunk files into a single shuffled output file using a heap-based external merge.
    
    This function uses a memory-efficient approach by reading only one line from each chunk file
    at a time and using a heap to maintain the order of random keys. The random keys are stripped
    from the output, producing a randomly shuffled version of the original lines.

    Args:
        chunk_files (list[str]): List of sorted temporary file paths, each containing
            lines prefixed with random keys.
        output_path (str): Path to the output file where merged content will be written.
            The output will contain only the original lines, without the random keys.

    Note:
        The input chunk files should be sorted by their random keys for the merge to work
        correctly. This is typically handled by tag_lines_with_random_keys().

    Example:
        >>> temp_chunks = ['/tmp/chunk1.txt', '/tmp/chunk2.txt']  # Files with tagged lines
        >>> merge_sorted_chunks(temp_chunks, 'output.txt')
    """
    def line_iter(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                key_str, content = line.split('\t', 1)
                yield (float(key_str), content)

    with open(output_path, 'w') as outfile:
        for _, line in heapq.merge(*(line_iter(fp) for fp in chunk_files)):
            outfile.write(line)

def shuffle_large_file_external(input_path: str, output_path: str) -> None:
    """
    Shuffles a large file by tagging each line with a random key and using an external merge sort.
    
    This function provides a memory-efficient way to randomly shuffle very large text files
    that don't fit in memory. It processes the file in chunks and uses temporary storage
    for intermediate results.

    Args:
        input_path (str): Path to the input file to be shuffled.
        output_path (str): Path where the shuffled output will be written.

    Example:
        >>> # Shuffle a 10GB log file
        >>> shuffle_large_file_external('huge_log.txt', 'shuffled_log.txt')
        
    Note:
        - The function creates temporary files that are automatically cleaned up
        - Memory usage is controlled by the chunk size, not the input file size
        - The shuffle is uniformly random regardless of the input file size
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_files = tag_lines_with_random_keys(input_path, temp_dir)
        merge_sorted_chunks(chunk_files, output_path)

def main() -> None:
    """
    Main function to parse command-line arguments and shuffle a large input file.

    This function provides a command-line interface to the file shuffling utility.
    It parses arguments for input file, output file, and optional chunk size.

    Example:
        $ python shuffle.py input.txt output.txt --lines-per-chunk 50000

    Command-line Arguments:
        input_path: Path to the input text file
        output_path: Path to the output shuffled file
        --lines-per-chunk: Number of lines to process per chunk (default: 100000)
    """
    parser = argparse.ArgumentParser(description="Shuffle a large file using external sorting.")
    parser.add_argument("input_path", help="Path to the input text file.")
    parser.add_argument("output_path", help="Path to the output shuffled file.")
    parser.add_argument("--lines-per-chunk", type=int, default=100000, help="Number of lines per chunk (default: 100000).")
    args = parser.parse_args()

    shuffle_large_file_external(args.input_path, args.output_path)

if __name__ == "__main__":
    main()
