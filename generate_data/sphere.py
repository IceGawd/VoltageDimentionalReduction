import numpy as np
import pandas as pd


# Parameters
N_SPHERE = 3000  # number of points
SEED_SPHERE = 123
np.random.seed(SEED_SPHERE)
filename="sphere_3k.csv"

# Generate points from a 3D standard normal and normalize to unit length
X_raw = np.random.randn(N_SPHERE, 3)
X_sphere = X_raw / np.linalg.norm(X_raw, axis=1, keepdims=True)
x, y, z = X_sphere[:,0], X_sphere[:,1], X_sphere[:,2]

# Save to CSV
label = (z * 10).astype(int) % 10  # Example label based on z-coordinate
df = pd.DataFrame({'label': label.astype(str), 'dx': x, 'dy': y, 'dz': z})
df.to_csv(filename, index=False)
print(f"Saved {N_SPHERE} points on a sphere to {filename}")