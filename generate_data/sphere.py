import argparse
import numpy as np
import pandas as pd

def sample_n_sphere(n_points, n_dim, filled=True, seed=None):
	rng = np.random.default_rng(seed)
	# Sample from standard normal
	X = rng.normal(size=(n_points, n_dim))
	# Normalize to unit sphere
	X /= np.linalg.norm(X, axis=1, keepdims=True)

	if filled:
		# Sample radius with proper distribution for uniform fill
		r = rng.random(n_points) ** (1.0 / n_dim)
		X *= r[:, None]

	return X

def main():
	parser = argparse.ArgumentParser(description="Generate an n-dimensional sphere dataset and save to CSV")
	parser.add_argument("--points", type=int, default=1000,
						help="Number of points to generate")
	parser.add_argument("--dim", type=int, default=3,
						help="Number of dimensions of the sphere")
	parser.add_argument("--seed", type=int, default=None,
						help="Random seed")
	parser.add_argument("--output", type=str, default="../../Voltage_Data/synthetic/n_sphere.csv",
						help="Output CSV file path")
	parser.add_argument("--filled", action="store_true",
						help="If set, sample uniformly inside the sphere instead of just the surface")
	parser.add_argument("--add_random_noise", type=float, default=0.0,
						help="Extra uniform noise to add to each coordinate")

	args = parser.parse_args()

	X = sample_n_sphere(args.points, args.dim, filled=args.filled, seed=args.seed)

	if args.add_random_noise > 0:
		X += np.random.uniform(-args.add_random_noise, args.add_random_noise, X.shape)

	# Build dataframe with dim1, dim2, ..., dimN
	col_names = [f"dim{i+1}" for i in range(args.dim)]
	df = pd.DataFrame(X, columns=col_names)
	df["label"] = X[:,args.dim-1]

	df.to_csv(args.output, index=False)
	print(f"Saved {args.dim}D sphere dataset with {args.points} points to {args.output} (filled={args.filled})")

if __name__ == "__main__":
	main()
