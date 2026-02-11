# -*- coding: utf-8 -*-


import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
from tqdm import tqdm
import warnings
 
warnings.filterwarnings("ignore")


# extract positions of non-H atoms
def alignDcd2Ref(psf_file_path, dcd_file_path,\
                 reference_pdb_file, aligned_file, aligned_npz,\
                 selection='not (resname WAT) and not (resname Na+) and not (name H*)'):
    
    """
    Align conformations in a dcd trajectory to a reference conformation
    
    ------------------------------Input------------------------------
    
    psf_file_path: str, topology file
    
    dcd_file_path: str, dcd file
    
    reference_pdb_file: str, pdb file of the reference conformation
    
    aligned_file: str, storage path of the aligned dcd
    
    aligned_npz: str, storage path of the aligned numpy array
    
    selection: str, the MDAnalysis atom selection language, which should be properly set to select non-H atoms.
               For details in the selection language: https://userguide.mdanalysis.org/stable/selections.html
    
    In our examples,
    For chignolin: selection = 'not (resname WAT) and not (resname Na+) and not (name H*)'
    
    ------------------------------Output------------------------------
    
    Carte_coor: float32, numpy array of aligned conformations
    
    """
         
    reference = mda.Universe(psf_file_path, reference_pdb_file)
    dcd_to_align = mda.Universe(psf_file_path,dcd_file_path)
    
    align.AlignTraj(dcd_to_align,  # trajectory to align
                    reference,  # reference
                    select=selection,  # selection of atoms to align
                    filename=aligned_file,  # file to write the trajectory to
                    match_atoms=True,  # whether to match atoms based on mass
                    ).run()
    
    # open the aligned DCD file
    dcd_aligned = mda.Universe(psf_file_path,aligned_file)
    psf_temp = mda.Universe(psf_file_path)
    
    position_list = []
    
    # directly traverse dcd_aligned.trajectory will cause an error
    for i in tqdm(dcd_aligned.trajectory):
        psf_atom = psf_temp.load_new(i.positions)
        selected_atom = psf_atom.select_atoms(selection)
        
        position = selected_atom.positions
        position_list.append(position[None,...].copy())
        
    Carte_coor = np.vstack(position_list)
    np.savez(aligned_npz, position=Carte_coor)
    
    return Carte_coor
