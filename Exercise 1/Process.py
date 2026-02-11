# -*- coding: utf-8 -*-

import MDAnalysis as mda
import numpy as np

# 1. Load PSF topology and trajectory
u = mda.Universe("tripeptide_nowater.psf", "tripeptide_nowater.dcd")

# 2. Define dihedral angle atom indices (use actual atom numbers)
dihedral_atoms = [
    [0, 1, 2, 3],      # dihedral 1 (atom indices start from 0)
    [4, 5, 6, 7],      # dihedral 2
    [8, 9, 10, 11],    # dihedral 3
    [12, 13, 14, 15],  # dihedral 4
    [16, 17, 18, 19],  # dihedral 5
    [20, 21, 22, 23],  # dihedral 6
]

# 3. Define phi angle atom indices (for columns 8-10)
phi_atoms = [
    [0, 1, 2, 3],    # phi 1
    [4, 5, 6, 7],    # phi 2
    [8, 9, 10, 11],  # phi 3
]

# 4. Calculate dihedrals and phi angles
results = []

for frame_idx, ts in enumerate(u.trajectory, start=1):
    row = [frame_idx]  # Column 1: frame number
    
    # Columns 2-7: sine (or cosine) values of 6 dihedral angles
    for atoms in dihedral_atoms:
        ag = u.atoms[atoms]
        angle_rad = mda.lib.distances.calc_dihedrals(
            ag[0].position, ag[1].position,
            ag[2].position, ag[3].position
        )
        row.append(np.sin(angle_rad))  # Use np.cos(angle_rad) if cosine is needed
    
    # Columns 8-10: phi angle values in degrees
    for atoms in phi_atoms:
        ag = u.atoms[atoms]
        phi_rad = mda.lib.distances.calc_dihedrals(
            ag[0].position, ag[1].position,
            ag[2].position, ag[3].position
        )
        row.append(np.degrees(phi_rad))
    
    results.append(row)

# 5. Save to npz file
tri_sin_phi = np.array(results)
np.savez_compressed("tri_sin_phi_data.npz", tri_sin_phi=tri_sin_phi)

print(f"✓ Generated data for {tri_sin_phi.shape[0]} frames")
print(f"✓ Array shape: {tri_sin_phi.shape}")