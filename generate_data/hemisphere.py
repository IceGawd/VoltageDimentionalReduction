
# Generate the same 4D hemisphere points and save to CSV
import pandas as pd
import numpy as np

# Use the same parameters as the previous cell
N_HEMI = 10000
SEED_HEMI = 789
np.random.seed(SEED_HEMI)

# Generate points on a 4D hemisphere by first generating points on a 4D sphere
# then keeping only those with positive 4th coordinate
X_raw = np.random.randn(N_HEMI * 2, 4)  # Generate extra points since we'll filter
X_sphere_4d = X_raw / np.linalg.norm(X_raw, axis=1, keepdims=True)

# Keep only points where the 4th coordinate is positive (hemisphere)
hemisphere_mask = X_sphere_4d[:, 3] >= 0
X_hemisphere_4d = X_sphere_4d[hemisphere_mask][:N_HEMI]  # Take first N_HEMI points

# If we don't have enough points, generate more
while X_hemisphere_4d.shape[0] < N_HEMI:
    X_raw_extra = np.random.randn(N_HEMI, 4)
    X_sphere_extra = X_raw_extra / np.linalg.norm(X_raw_extra, axis=1, keepdims=True)
    hemisphere_mask_extra = X_sphere_extra[:, 3] >= 0
    X_hemisphere_extra = X_sphere_extra[hemisphere_mask_extra]
    X_hemisphere_4d = np.vstack([X_hemisphere_4d, X_hemisphere_extra])

X_hemisphere_4d = X_hemisphere_4d[:N_HEMI]  # Ensure exactly N_HEMI points

# Embed in R^100 by placing the 4D hemisphere in the first 4 dimensions
# and adding small noise to the remaining 96 dimensions
noise_dims = np.random.normal(0, 0.01, (N_HEMI, 96))  # Small noise in other dimensions
label = ((X_hemisphere_4d[:, 3] + 1) * 4).astype(int) % 10
X_hemi_100d = np.hstack([X_hemisphere_4d, noise_dims])

# Create DataFrame with descriptive column names
# First 4 columns are the hemisphere coordinates, rest are noise dimensions
column_names = ['d1', 'd2', 'd3', 'd4'] + [f'd_noise_{i+1}' for i in range(96)]
df_hemisphere = pd.DataFrame(X_hemi_100d, columns=column_names)

# Add a column for label (4th dimension)
df_hemisphere['label'] = label

# Save to CSV
csv_filename = '4d_hemisphere_points.csv'
df_hemisphere.to_csv(csv_filename, index=False)

print(f"Generated {N_HEMI} points on 4D hemisphere embedded in R^100")
print(f"Saved to: {csv_filename}")
print(f"DataFrame shape: {df_hemisphere.shape}")
print(f"4th coordinate range: [{df_hemisphere['d4'].min():.3f}, {df_hemisphere['d4'].max():.3f}]")
print(f"All 4th coordinates >= 0: {(df_hemisphere['d4'] >= 0).all()}")
print(f"Mean distance from origin (4D): {np.mean(np.linalg.norm(X_hemisphere_4d, axis=1)):.6f}")

# Display first few rows
print("\nFirst 5 rows:")
print(df_hemisphere.head())


