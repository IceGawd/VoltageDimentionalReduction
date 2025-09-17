import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_swiss_roll

def main():
	parser = argparse.ArgumentParser(description="Generate a Swiss Roll dataset and save to CSV")
	parser.add_argument("--points", type=int, default=1000,
						help="Number of points to generate")
	parser.add_argument("--noise", type=float, default=0.0,
						help="Standard deviation of Gaussian noise to add")
	parser.add_argument("--seed", type=int, default=None,
						help="Starting random seed")
	parser.add_argument("--output", type=str, default="../../Voltage_Data/synthetic/swiss_roll.csv",
						help="Output CSV file path")
	parser.add_argument("--add_random_noise", type=float, default=0.0,
						help="Extra uniform noise to add to each coordinate")

	args = parser.parse_args()

	X, t = make_swiss_roll(n_samples=args.points, noise=args.noise, random_state=args.seed)

	if args.add_random_noise > 0:
		X += np.random.uniform(-args.add_random_noise, args.add_random_noise, X.shape)

	df = pd.DataFrame(X, columns=["x", "y", "z"])
	df["t"] = t  # manifold parameter

	df.to_csv(args.output, index=False)
	print(f"Saved swiss roll dataset with {args.points} points to {args.output}")

if __name__ == "__main__":
	main()
