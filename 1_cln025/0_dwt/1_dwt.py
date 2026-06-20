import pywt
import numpy as np
import pandas as pd

data = np.load('../0_Align/cln025_aligned.npz')
print("Keys in npz:", data.files)
print("Shape of position data:", data['position'].shape)

dcd_position = data['position'].copy()
data.close()

# Reshape the data for wavelet transform
dcd_position = dcd_position.reshape((dcd_position.shape[0], -1))

# Initialize the wavelet-transformed data array
wt_dcdposition = np.zeros_like(dcd_position)

# Apply wavelet decomposition and reconstruction the data
for i in range(dcd_position.shape[1]):
    # Perform the wavelet decomposition
    coeffs = pywt.wavedec(dcd_position[:, i], 'coif7', level=4, mode='periodization')
    # Zero out all but the approximation coefficients
    for j in range(1, 5):
        coeffs[-j] = np.zeros_like(coeffs[-j])
    # Reconstruct the signal using only the approximation coefficients
    reconstructed = pywt.waverec(coeffs, 'coif7', mode='periodization')
    wt_dcdposition[:, i] = reconstructed[:dcd_position.shape[0]]

# Save the wavelet-transformed data to a new file
Carte_coor = wt_dcdposition.reshape((wt_dcdposition.shape[0], 166, 3))   
np.savez('./wt_aligned.npz', position=Carte_coor)

wt_data = np.load('./wt_aligned.npz')
print("Keys in wt_aligned_npz:", wt_data.files)

pos = wt_data['position']
print("Shape of position wt_data:", pos.shape)

import MDAnalysis as mda
from MDAnalysis.coordinates.DCD import DCDReader, DCDWriter
from MDAnalysis.coordinates.memory import MemoryReader
import numpy as np

# Define file paths
psf_file = '../0_Align/cln025.psf'
wt_npz_file = './wt_aligned.npz'
original_dcd = '../0_Align/cln025_aligned.dcd'

u_orig = mda.Universe(psf_file, original_dcd)
dt = u_orig.trajectory.dt
print(f"Original trajectory timestep: {dt} fs")

# Load the PSF file and the NPZ file
u = mda.Universe(psf_file)
wt_data = np.load(wt_npz_file)

# Extract positions from the NPZ file
positions = wt_data['position']
n_frames, n_atoms, n_coords = positions.shape

# Validate the number of atoms
assert u.atoms.n_atoms == n_atoms, "Number of atoms in PSF and NPZ do not match."

mem_reader = MemoryReader(positions, dt=dt)
u.trajectory = mem_reader

# Write the positions to a new DCD file
with DCDWriter('wt_aligned.dcd', n_atoms) as W:
    for ts in u.trajectory:
        W.write(u)

import MDAnalysis as mda
import numpy as np

psf_file = '../0_Align/cln025.psf'
wt_npz_file = './wt_aligned.npz'

u = mda.Universe(psf_file)
print("Number of atoms in PSF:", u.atoms.n_atoms)

wt_data = np.load(wt_npz_file)
positions = wt_data['position']
print("Number of atoms in NPZ:", positions.shape[1])



