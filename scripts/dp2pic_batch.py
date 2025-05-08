import os
import pickle
import numpy as np
import dpdata

element_to_atomic_number = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
    'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

def convert_deepmd_to_custom(deepmd_dir, output_path):
    subdirs = sorted([d for d in os.listdir(deepmd_dir) if os.path.isdir(os.path.join(deepmd_dir, d))])
    
    for subdir in subdirs:

        subdir_path = os.path.join(deepmd_dir, subdir)

        # Load the DeepMD data
        data = dpdata.LabeledSystem(subdir_path, fmt="deepmd/npy")

        # Extract relevant data
        E = data['energies']
        F = data['forces']
        R = data['coords']
        Z = data['atom_types']
        C = data['cells']
        N = len(data['atom_types'])
        N = np.full(len(E), N)
        S = 0 - data["virials"] 
        for i in range(len(Z)):
            Z[i] = element_to_atomic_number[data['atom_names'][Z[i]]]
        Z = np.repeat(Z[np.newaxis, :], len(E), axis=0)

        print(f"Processing subdir: {subdir}")
        print(E, F.shape, R.shape, Z.shape, C.shape, N)

        # Create the output directory structure
        output_dir = os.path.join("../dataset", os.path.basename(output_path))
        raw_dir = os.path.join(output_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # Save the data as a pickle file in the raw directory
        output_file = os.path.join(raw_dir, f"{os.path.basename(output_path)}_{subdir}.pickle")
        with open(output_file, 'wb') as f:
            pickle.dump({'E': E, 'F': F, 'R': R, 'z': Z, 'cell': C, 'natoms': N, "stress": S}, f)

        print(f"Conversion for {subdir} completed. Data saved to {output_file}")

# Example usage
deepmd_dir = "t1_perov/data"
output_path = "t1"
convert_deepmd_to_custom(deepmd_dir, output_path)

