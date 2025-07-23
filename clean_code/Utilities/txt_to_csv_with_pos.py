import spacy
import csv

# Load English POS tagger
nlp = spacy.load("en_core_web_sm")

# Input and output files
input_file = r"..\..\Voltage_Data\glove\shuffled_output.txt"
output_file = r"..\..\Voltage_Data\glove\glove_with_pos.csv"

def process_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as fin:
        # Peek at first valid line to get vector length
        for line in fin:
            parts = line.strip().split()
            if len(parts) >= 2:
                vector_length = len(parts) - 1
                break
        else:
            print("No valid lines in input file.")
            return

        # Reset file pointer to beginning
        fin.seek(0)

        with open(output_file, "w", encoding="utf-8", newline='') as fout:
            writer = csv.writer(fout)

            # Write header: POS, dim1, ..., dimN, word
            header = ["POS"] + [f"dim{i+1}" for i in range(vector_length)] + ["word"]
            writer.writerow(header)

            for line in fin:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                word = parts[0]
                vec = parts[1:]

                try:
                    doc = nlp(word)
                    pos = doc[0].pos_ if doc else "X"
                except Exception:
                    pos = "X"

                writer.writerow([pos] + vec + [f'"{word}"'])

    print(f"Finished writing to {output_file}")

if __name__ == "__main__":
    process_file(input_file, output_file)
