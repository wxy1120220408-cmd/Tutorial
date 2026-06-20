# -*- coding: utf-8 -*-

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Function to align DCD trajectory to a reference structure and extract positions of non-H atoms
def alignDcd2Ref(psf_file_path, dcd_file_path, reference_pdb_file, aligned_file, aligned_npz, selection):
             
    reference = mda.Universe(psf_file_path, reference_pdb_file)
    dcd_to_align = mda.Universe(psf_file_path, dcd_file_path)
    
    align.AlignTraj(dcd_to_align,  # trajectory to align
                    reference,  # reference
                    select=selection,  # selection of atoms to align
                    filename=aligned_file,  # file to write the trajectory to
                    match_atoms=True  # whether to match atoms based on mass
                   ).run()
    
    # Open the aligned DCD file
    dcd_aligned = mda.Universe(psf_file_path, aligned_file)
    psf_temp = mda.Universe(psf_file_path)
    
    position_list = []
    
    # Traverse dcd_aligned.trajectory
    for i in tqdm(dcd_aligned.trajectory):
        psf_atom = psf_temp.load_new(i.positions)
        selected_atom = psf_atom.select_atoms(selection)
        
        position = selected_atom.positions
        position_list.append(position[None, ...].copy())
        
    Carte_coor = np.vstack(position_list)
    np.savez(aligned_npz, position=Carte_coor)
    
    return Carte_coor

# Define file paths and parameters
psf_file_path = './cln025.psf'
dcd_file_path = './cln025.dcd'
reference_pdb_file = './cln025.pdb'
aligned_file = './cln025_aligned.dcd'
aligned_npz = './cln025_aligned.npz'
# selection = 'segid P and not name H*'
# selection = "chainID P and not name H*"
selection = "protein"

# Run the alignment and extraction process
alignDcd2Ref(psf_file_path, dcd_file_path, reference_pdb_file, aligned_file, aligned_npz, selection)
