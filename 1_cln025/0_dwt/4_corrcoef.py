import numpy as np
import pandas as pd

# 1. Load Data
filename = './COLVAR-dwt-filter'
chunk_size = 1024 * 256  

# Read the first line first to get correct column names
with open(filename, "r") as f:
    for line in f:
        if line.startswith("#! FIELDS"):
            header = line.strip().split()[2:]  # Skip "#! FIELDS"
            break

# Initialize variable to store results
all_data = []

# Read file in chunks and process
for chunk in pd.read_csv(filename, delim_whitespace=True, comment='#', names=header, skiprows=1, chunksize=chunk_size):
    # Append each chunk to the list
    all_data.append(chunk.astype(np.float32))

# Concatenate data
df = pd.concat(all_data, ignore_index=True)

# Remove the first column 'time'
target_cols = header[1:]
dist = df[target_cols]
print(f"Data shape: {dist.shape}")

def select_low_corr_features(data, corr_threshold=0.9, save_path=None):
    """
    Filter features based on correlation and variance:
    1. Prioritize retaining features with higher variance.
    2. Remove features with correlation >= corr_threshold with already retained features.

    Parameters:
    - data: DataFrame, original data
    - corr_threshold: float, correlation threshold, default 0.9
    - save_path: str, file path to save results, e.g., "seldist.csv"

    Returns:
    - selected_cols: list, names of filtered features
    """
    # Calculate correlation matrix
    coef = data.corr().abs()
    np.fill_diagonal(coef.values, 0)

    # Sort features by variance in descending order
    variances = data.var()
    sorted_cols = variances.sort_values(ascending=False).index

    # Save column names sorted by variance to var.csv
    with open("var.csv", "w") as f:
        for col in sorted_cols:
            f.write(f"{col}\n")

    # Initialize keep mask
    keep = pd.Series(True, index=coef.columns)

    # Iteratively filter features
    for col in sorted_cols:
        if keep[col]:
            # Find features highly correlated with the current feature
            to_drop = coef.index[coef[col] >= corr_threshold]
            # Mark them for removal
            keep[to_drop] = False
            # Ensure the current feature is kept
            keep[col] = True

    # Get the list of retained features
    selected_cols = keep[keep].index.tolist()

    # Optional save
    if save_path:
        with open(save_path, "w") as f:
            for col in selected_cols:
                f.write(f"{col}\n")

    return selected_cols


# Execution
selected_cols = select_low_corr_features(
    dist, 
    corr_threshold=0.9, 
    save_path="seldist.csv")


print(f"Number of features retained: {len(selected_cols)}")


# 3. Generate plumed-seldist.dat

df = pd.read_csv("seldist.csv", header=None, names=['feature'])

with open("plumed-seldist.dat", "w") as f:
    f.write("# vim: ft=plumed\n\n")
    f.write("####################################\n")
    f.write("#       >> Chignolin <<\n")
    f.write("#    DRIVER - correcoef_filter\n")
    f.write("####################################\n\n")
    f.write("UNITS LENGTH=A\n\n")

    for feature in df['feature']:
        # Assuming format is like "dist_1_2", split to get atom indices
        atoms = feature.split('_')[1:]
        atom_str = ",".join([str(int(a)) for a in atoms])
        f.write(f"{feature}: DISTANCE ATOMS={atom_str}\n")

    f.write("\nPRINT STRIDE=25 ARG=* FILE=COLVAR-seldist\n")