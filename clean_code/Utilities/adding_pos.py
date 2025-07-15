"""
pos_tag_glove.py

This script reads a GloVe-style word embedding file, uses spaCy to extract the part of speech (POS)
for each word, and writes a new file in the format:
    POS val1 val2 ... valN  # word

This allows POS to be used as a label while preserving the original word as a comment.
"""

import spacy

# Load English POS tagger
nlp = spacy.load("en_core_web_sm")

# Input and output files
input_file = r"..\..\Voltage_Data\glove\shuffled_output.txt"
output_file = r"..\..\Voltage_Temp\glove_with_pos.txt"

def process_file(input_file,output_file):
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