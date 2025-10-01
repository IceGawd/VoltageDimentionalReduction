import argparse
import numpy as np
import pandas as pd

def sample_torus(n_points, R=2.0, r=1.0, seed=None):
	rng = np.random.default_rng(seed)
	theta = rng.uniform(0, 2*np.pi, n_points)  # around big circle
	phi = rng.uniform(0, 2*np.pi, n_points)    # around tube

	x = (R + r * np.cos(phi)) * np.cos(theta)
	y = (R + r * np.cos(phi)) * np.sin(theta)
	z = r * np.sin(phi)

	return np.vstack([x, y, z]).T, theta, phi

def main():
	parser = argparse.ArgumentParser(description="Generate a 3D torus dataset and save to CSV")
	parser.add_argument("--points", type=int, default=1000,
						help="Number of points to generate")
	parser.add_argument("--seed", type=int, default=None,
						help="Random seed")
	parser.add_argument("--output", type=str, default="../../Voltage_Data/synthetic/torus.csv",
						help="Output CSV file path")
	parser.add_argument("--R", type=float, default=2.0,
						help="Major radius of the torus")
	parser.add_argument("--r", type=float, default=1.0,
						help="Minor radius of the torus")
	parser.add_argument("--add_random_noise", type=float, default=0.0,
						help="Extra uniform noise to add to each coordinate")

	args = parser.parse_args()

	X, theta, phi = sample_torus(args.points, R=args.R, r=args.r, seed=args.seed)

	if args.add_random_noise > 0:
		X += np.random.uniform(-args.add_random_noise, args.add_random_noise, X.shape)

	df = pd.DataFrame(X, columns=["dimx", "dimy", "dimz"])
	df["label"] = theta

	df.to_csv(args.output, index=False)
	print(f"Saved torus dataset with {args.points} points to {args.output} (R={args.R}, r={args.r})")

if __name__ == "__main__":
	main()
