# a small program to generate points in [-1,+1]^d.
# there are two modes:
# 1. generate a grid of points in [-1,+1]^d, the number of points in each dimension is given by the parameter `grid_size`.
# 2. generate random points in [-1,+1]^d, the number of points is given by the parameter `num_points`.
#
# the points are output in a cvs file with the following columns:
# label, x1, x2, ..., xd, weight
# where label is the integer whose binary representation corresponds to whether x_i is negative or positive (0 for negative, 1 for positive), x_i is the i-th coordinate of the point, 
# # and weight=1.0 is the weight of the point.
import numpy as np
def generate_grid_points(grid_size: int, d: int) -> np.ndarray:
    """
    Generates a grid of points in [-1, +1]^d.

    Args:
        grid_size (int): Number of points in each dimension.
        d (int): Dimension of the space.

    Returns:
        np.ndarray: Array of shape (grid_size^d, d) containing the grid points.
    """
    ranges = [np.linspace(-1, 1, grid_size)] * d
    grid_points = np.array(np.meshgrid(*ranges)).T.reshape(-1, d)
    return grid_points
def generate_random_points(num_points: int, d: int) -> np.ndarray:
    """
    Generates random points in [-1, +1]^d.

    Args:
        num_points (int): Number of points to generate.
        d (int): Dimension of the space.

    Returns:
        np.ndarray: Array of shape (num_points, d) containing the random points.
    """
    return np.random.uniform(-1, 1, (num_points, d))
def generate_points(grid_size: int = 10, num_points: int = 1000, d: int = 2, mode: str = 'grid') -> np.ndarray:
    """
    Generates points in [-1, +1]^d based on the specified mode.
    Args:
        grid_size (int): Number of points in each dimension for grid mode.
        num_points (int): Number of random points for random mode.
        d (int): Dimension of the space.
        mode (str): 'grid' for grid points, 'random' for random points.
    Returns:
        np.ndarray: Array of shape (N, d) containing the generated points.
    """ 
    if mode == 'grid':
        return generate_grid_points(grid_size, d)
    elif mode == 'random':
        return generate_random_points(num_points, d)
    else:
        raise ValueError("Mode must be either 'grid' or 'random'.")
if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Generate points in [-1, +1]^d.")
    parser.add_argument('--output', type=str, default='generated_points.csv', help='Output CSV file name.')
    parser.add_argument('--grid_size', type=int, default=10, help='Number of points in each dimension for grid mode.')
    parser.add_argument('--num_points', type=int, default=1000, help='Number of random points for random mode.')
    parser.add_argument('--d', type=int, default=2, help='Dimension of the space.')
    parser.add_argument('--mode', type=str, choices=['grid', 'random'], default='grid', help='Mode of point generation.')

    args = parser.parse_args()

    points = generate_points(args.grid_size, args.num_points, args.d, args.mode)
    
    # Create labels and weights
    labels = np.array([int(''.join(['1' if x >= 0 else '0' for x in point]), 2) for point in points])
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(points, columns=[f'x{i+1}' for i in range(args.d)])
    df['label'] = labels
    df=df[['label'] + [f'x{i+1}' for i in range(args.d)]]
    df.to_csv(args.output, index=False)
    print(f"{args.output}: {points.shape[0]} points in {args.mode} mode with dimension {args.d}.")
    
