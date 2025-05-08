import os
import pickle
import numpy as np
from ase.io import read
from tqdm import tqdm

def convert_xyz_to_custom(xyz_file, output_path, chunk_size=100000):
    # Load the XYZ file
    atoms = read(xyz_file, index=":")
    total_frames = len(atoms)

    # Initialize lists
    E, F, R, Z, C, N, S = [], [], [], [], [], [], []

    # Create the output directory structure
    output_dir = os.path.join("../dataset", os.path.basename(output_path))
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Process data with progress bar
    for i, atoms_obj in enumerate(tqdm(atoms, total=total_frames, desc="Processing frames")):
        E.append(atoms_obj.get_potential_energy())
        F.append(atoms_obj.get_forces().tolist())
        R.append(atoms_obj.get_positions().tolist())
        Z.append([atom.number for atom in atoms_obj])
        C.append(atoms_obj.get_cell().tolist())
        N.append(len(atoms_obj.get_forces()))
        S.append(atoms_obj.get_stress().tolist())

        # Save chunk when reaching chunk_size
        if (i + 1) % chunk_size == 0 or i == total_frames - 1:
            chunk_number = i // chunk_size + 1
            save_chunk(E, F, R, Z, C, N, S, raw_dir, output_path, chunk_number)
            # Clear lists after saving
            E, F, R, Z, C, N, S = [], [], [], [], [], [], []

    print(f"Conversion completed. Data saved in chunks to {raw_dir}")

def save_chunk(E, F, R, Z, C, N, S, raw_dir, output_path, chunk_number):
    output_file = os.path.join(raw_dir, f"{os.path.basename(output_path)}_chunk{chunk_number}.pickle")
    with open(output_file, 'wb') as f:
        pickle.dump({
            'E': np.array(E),
            'F': F,
            'R': R,
            'z': Z,
            'cell': np.array(C),
            'natoms': np.array(N),
            'stress': np.array(S)
        }, f)
    print(f"Chunk {chunk_number} saved to {output_file}")

# Example usage
xyz_file = "train.extxyz"
output_path = "MP_train"
convert_xyz_to_custom(xyz_file, output_path)

