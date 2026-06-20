from Bio.PDB import PDBParser
import itertools
import numpy as np
import os

# Define file paths
pdb_file = "./cln025.pdb"
output_file = "./plumed-distance.dat"

# Ensure output directory exists (optional, helps prevent errors if folder is missing)
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Read PDB file
# Note: You might need to handle the case where the PDB file doesn't exist
if not os.path.exists(pdb_file):
    print(f"Error: PDB file not found at {pdb_file}")
else:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_file)

    # Extract all non-hydrogen atoms (record serial number and coordinates)
    atoms = []
    for atom in structure.get_atoms():
        if not atom.element.upper().startswith("H"):  # Ignore hydrogen
            atoms.append((atom.serial_number, atom.coord))

    # 2. Get bonded atom pairs (determined from CONECT records)
    bonded_pairs = set()
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("CONECT"):
                nums = list(map(int, line.split()[1:]))
                atom1 = nums[0]
                # CONECT records list bonded atoms; add pairs to set
                for atom2 in nums[1:]:
                    bonded_pairs.add(tuple(sorted((atom1, atom2))))

    # 3. Generate non-bonded atom pairs and write to file
    with open(output_file, "w") as f:
        f.write("# vim: ft=plumed\n\n")
        f.write("####################################\n")
        f.write("#      >> Chignolin <<\n")
        f.write("#  DRIVER - Compute descriptors\n")
        f.write("####################################\n\n")
        f.write("UNITS LENGTH=A\n\n")

        # Iterate through all combinations of 2 atoms
        for (n1, _), (n2, _) in itertools.combinations(atoms, 2):
            # Check if the pair is NOT in the bonded list
            if tuple(sorted((n1, n2))) not in bonded_pairs:
                label = f"dd_{n1:03d}_{n2:03d}"
                f.write(f"{label}: DISTANCE ATOMS={n1},{n2}\n")
        
        f.write("\n")
        f.write("PRINT STRIDE=1 ARG=* FILE=COLVAR-dwt\n")

    print(f"Generated {output_file}")