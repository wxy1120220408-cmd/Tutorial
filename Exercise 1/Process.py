# -*- coding: utf-8 -*-
"""
Preprocessing Script for MD Trajectory Analysis
================================================
This script demonstrates how to extract collective variables from MD trajectories
using MDAnalysis. It supports multiple trajectory formats and can be easily adapted
for different collective variable types.
"""

import MDAnalysis as mda
import numpy as np

# ============================================================================
# SECTION 1: Load Trajectory Files
# ============================================================================
# MDAnalysis provides a unified interface for reading different trajectory formats.
# Simply change the file extensions below according to your MD software:

# CURRENT SETUP: NAMD/CHARMM format
u = mda.Universe("tripeptide_nowater.psf",      # Topology file
                 "tripeptide_nowater.dcd")      # Trajectory file

# --- ALTERNATIVE FORMATS (uncomment and modify as needed) ---
# 
# For GROMACS:
# u = mda.Universe("system.gro", "trajectory.xtc")
# u = mda.Universe("system.tpr", "trajectory.trr")
#
# For AMBER:
# u = mda.Universe("system.prmtop", "trajectory.nc")
# u = mda.Universe("system.prmtop", "trajectory.mdcrd")
#
# For OpenMM:
# u = mda.Universe("system.pdb", "trajectory.dcd")
# u = mda.Universe("system.pdb", "trajectory.h5")
#
# For LAMMPS:
# u = mda.Universe("system.data", "trajectory.lammpstrj", format="LAMMPSDUMP")
#
# For PDB files (multiple frames):
# u = mda.Universe("trajectory.pdb")

print(f"✓ Loaded {len(u.trajectory)} frames")
print(f"✓ System contains {len(u.atoms)} atoms")

# ============================================================================
# SECTION 2: Define Collective Variables
# ============================================================================
# This tutorial uses DIHEDRAL ANGLES as collective variables for alanine tripeptide.
# For other systems or collective variable types, modify this section accordingly.

# --- CURRENT SETUP: Backbone Dihedral Angles ---
# Define atom indices for 6 dihedral angles (phi and psi angles)
# IMPORTANT: Atom indices start from 0 in MDAnalysis
dihedral_atoms = [
    [0, 1, 2, 3],      # dihedral 1 (e.g., phi_1)
    [4, 5, 6, 7],      # dihedral 2 (e.g., psi_1)
    [8, 9, 10, 11],    # dihedral 3 (e.g., phi_2)
    [12, 13, 14, 15],  # dihedral 4 (e.g., psi_2)
    [16, 17, 18, 19],  # dihedral 5 (e.g., phi_3)
    [20, 21, 22, 23],  # dihedral 6 (e.g., psi_3)
]

# Additional phi angles for detailed analysis (columns 8-10 in output)
phi_atoms = [
    [0, 1, 2, 3],    # phi 1
    [4, 5, 6, 7],    # phi 2
    [8, 9, 10, 11],  # phi 3
]

# --- ALTERNATIVE COLLECTIVE VARIABLES (uncomment and modify as needed) ---
#
# A. ATOM-ATOM DISTANCES
# Example: Distance between CA atoms of residues 1 and 10
# Replace the calculation loop below with:
# """
# distances = []
# atom1 = u.select_atoms("resid 1 and name CA")[0]
# atom2 = u.select_atoms("resid 10 and name CA")[0]
# for ts in u.trajectory:
#     dist = mda.lib.distances.calc_bonds(atom1.position, atom2.position)
#     distances.append(dist)
# """

# B. RADIUS OF GYRATION
# Example: Rg of the entire protein
# """
# rg_values = []
# protein = u.select_atoms("protein")
# for ts in u.trajectory:
#     rg = protein.radius_of_gyration()
#     rg_values.append(rg)
# """

# C. RMSD (Root Mean Square Deviation)
# Example: RMSD relative to first frame
# """
# from MDAnalysis.analysis import rms
# R = rms.RMSD(u, u, select='backbone', ref_frame=0)
# R.run()
# rmsd_values = R.results.rmsd[:, 2]  # RMSD values
# """

# D. CONTACT MAPS
# Example: Number of contacts between two regions
# """
# from MDAnalysis.analysis import contacts
# region1 = u.select_atoms("resid 1-10")
# region2 = u.select_atoms("resid 20-30")
# contact_analysis = contacts.Contacts(u, select=(region1, region2), 
#                                      radius=4.5)
# contact_analysis.run()
# """

# ============================================================================
# SECTION 3: Calculate Collective Variables for Each Frame
# ============================================================================
results = []

for frame_idx, ts in enumerate(u.trajectory, start=1):
    row = [frame_idx]  # Column 1: frame number
    
    # --- Calculate 6 dihedral angles (columns 2-7) ---
    # We use sine transformation: sin(angle) to handle periodicity
    # NOTE: You can use cos(angle) instead, or even both sin and cos
    for atoms in dihedral_atoms:
        ag = u.atoms[atoms]
        angle_rad = mda.lib.distances.calc_dihedrals(
            ag[0].position, ag[1].position,
            ag[2].position, ag[3].position
        )
        row.append(np.sin(angle_rad))  
        # For cosine: use np.cos(angle_rad)
        # For raw angle: use np.degrees(angle_rad) or angle_rad
    
    # --- Calculate 3 phi angles in degrees (columns 8-10) ---
    for atoms in phi_atoms:
        ag = u.atoms[atoms]
        phi_rad = mda.lib.distances.calc_dihedrals(
            ag[0].position, ag[1].position,
            ag[2].position, ag[3].position
        )
        row.append(np.degrees(phi_rad))
    
    results.append(row)
    
    # Print progress every 10000 frames
    if frame_idx % 10000 == 0:
        print(f"  Processed {frame_idx} frames...")

# ============================================================================
# SECTION 4: Save Results
# ============================================================================
# Convert list to numpy array
tri_sin_phi = np.array(results)

# Save in compressed NPZ format (recommended for large datasets)
np.savez_compressed("tri_sin_phi_data.npz", tri_sin_phi=tri_sin_phi)

# --- ALTERNATIVE SAVE FORMATS ---
#
# A. Save as plain text (easier to inspect, but larger file size)
# np.savetxt("tri_sin_phi_data.txt", tri_sin_phi, fmt='%.6f')
#
# B. Save as CSV (compatible with Excel, Origin, etc.)
# np.savetxt("tri_sin_phi_data.csv", tri_sin_phi, delimiter=',', fmt='%.6f')
#
# C. Save as binary NPY (faster loading, single array only)
# np.save("tri_sin_phi_data.npy", tri_sin_phi)
