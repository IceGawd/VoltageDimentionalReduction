"""
Part of Speech Tagger for Word Embeddings

This script enhances GloVe-style word embeddings by adding part of speech (POS) tags.
It processes word embedding files and annotates each word with its grammatical role
using spaCy's natural language processing capabilities.

Input Format:
    word val1 val2 ... valN

Output Format:
    POS val1 val2 ... valN  # word

Where:
    - POS: Part of speech tag (e.g., NOUN, VERB, ADJ, etc.)
    - val1...valN: The original word embedding values
    - word: The original word (preserved as a comment)

Example:
    Input line:  "cat 0.123 0.456 0.789"
    Output line: "NOUN 0.123 0.456 0.789  # cat"

Note:
    Uses spaCy's 'en_core_web_sm' model for POS tagging.
    Words that cannot be tagged are marked with 'X'.
"""

import spacy
from typing import TextIO

# Load English POS tagger
nlp = spacy.load("en_core_web_sm")

# Input and output files
input_file = r"..\..\Voltage_Data\glove\shuffled_output.txt"
output_file = r"..\..\Voltage_Temp\glove_with_pos.txt"

def process_file(input_file: str, output_file: str) -> None:
    """
    Process a word embedding file to add POS tags to each word.

    This function reads a word embedding file line by line, extracts the word,
    determines its part of speech using spaCy, and writes the tagged version
    to a new file. The original word embedding values are preserved.

    Args:
        input_file (str): Path to the input word embedding file.
            Each line should be in the format: "word val1 val2 ... valN"
        output_file (str): Path where the tagged output will be written.
            Each line will be in the format: "POS val1 val2 ... valN  # word"

    Note:
        - Malformed lines (less than 2 parts) are skipped
        - If POS tagging fails, 'X' is used as the tag
        - Files are processed using UTF-8 encoding

    Example:
        >>> process_file("glove.txt", "glove_tagged.txt")
    """
    with open(input_file, "r", encoding="utf-8") as fin, \
        open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            parts = line.strip().split()
            if len(parts) < 2:
                continue  # Skip malformed lines

            word = parts[0]
            vec = parts[1:]

            try:
                doc = nlp(word)
                pos = doc[0].pos_ if doc else "X"
            except Exception:
                pos = "X"

            fout.write(f"{pos} {' '.join(vec)}  # {word}\n")

    print(f"Finished writing to {output_file}")

if __name__ == "__main__":
    process_file(input_file, output_file)