import sys
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from pywt import wavedec

# Config
filename = './COLVAR-dwt'
chunk_size = 256 * 1024
wavelet = 'coif7'
level = 4

# 1. Read Header
with open(filename, "r") as f:
    for line in f:
        if line.startswith("#! FIELDS"):
            header = line.strip().split()[2:]
            break

# 2. Read Data
print(f"Reading {filename}...")
chunks = []
for chunk in pd.read_csv(filename, sep='\s+', comment='#', names=header, skiprows=1, chunksize=chunk_size):
    chunks.append(chunk.astype(np.float32))

df = pd.concat(chunks, ignore_index=True)
np.savez("COLVAR-dwt.npz", data=df.values, header=header)

# 3. Wavelet Analysis
print("Analyzing...")
results = []
for col in tqdm(header[1:], desc="DWT"):
    cA = wavedec(df[col].values, wavelet, level=level)[0]
    results.append([col, np.sum(cA**2), np.std(cA)])

# 4. Process Results (Sort & Filter)
df_std = pd.DataFrame(results, columns=['feature', 'energy_cA', 'std_val'])
df_std.to_csv("COLVAR-dwt-std.csv", index=False)

df_sorted = df_std.sort_values('std_val', ascending=False).reset_index(drop=True)
df_sorted.to_csv("COLVAR-dwt-std-sorted.csv")

n_top = math.ceil(len(df_sorted) * 0.8)
df_top = df_sorted.iloc[:n_top]

# 5. Generate Plumed File
outfile = "plumed-dwt-filter.dat"
with open(outfile, "w") as f:
    f.write("# vim: ft=plumed\nUNITS LENGTH=A\n\n")
    for feat in df_top['feature']:
        # Parse 'dd_001_002' -> '1,2'
        atoms = ",".join(str(int(x)) for x in feat.split('_')[1:])
        f.write(f"{feat}: DISTANCE ATOMS={atoms}\n")
    f.write("\nPRINT STRIDE=25 ARG=* FILE=COLVAR-dwt-filter\n")

print(f"Done. Top 80% features ({len(df_top)}) written to {outfile}")