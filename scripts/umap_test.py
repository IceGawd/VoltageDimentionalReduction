import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import umap

def main():
	parser = argparse.ArgumentParser(description="UMAP visualization from CSV")
	parser.add_argument("csv", type=str, help="Path to input CSV file")
	parser.add_argument("--output", type=str, default="umap_plot.png", help="Output image file")
	args = parser.parse_args()

	# Load CSV
	df = pd.read_csv(args.csv)

	if "label" not in df.columns:
		raise ValueError("CSV must contain a 'label' column.")

	# Extract labels as floats
	labels = df["label"].astype(float).to_numpy()

	# Use all columns except label as features
	feature_cols = [c for c in df.columns if c != "label"]
	X = df[feature_cols].to_numpy()

	# Run UMAP
	umap_model = umap.UMAP(n_components=2)
	embedding = umap_model.fit_transform(X)

	# Plot
	fig, ax = plt.subplots(figsize=(10, 8))
	ax.set_facecolor('black')
	fig.patch.set_facecolor('black')
	scatter = plt.scatter(
		embedding[:, 0],
		embedding[:, 1],
		c=labels,
		cmap=cm.magma,
		s=5,
	)
	plt.savefig(args.output, dpi=300)
	print(f"Saved UMAP plot to {args.output}")

if __name__ == "__main__":
	main()
