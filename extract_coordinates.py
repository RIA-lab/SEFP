aa_map = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5,
              'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11,
              'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17,
              'TYR': 18, 'VAL': 19, 'OTHER': 20}


def extract_structure_coordinates(pdb_path):
    # Initialize arrays to store coordinates and one-hot encoding
    coordinates = []

    # Read PDB file and extract relevant information
    with open(pdb_path, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            if line.startswith('ATOM'):
                atom_name = line[13:16].strip()
                residue_name = line[17:20].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                # Check if the atom is a C_α atom and if the residue is in our list of amino acids
                if atom_name == 'CA':
                    aa_index = aa_map.get(residue_name, 20)
                    coordinates.append([x, y, z] + [aa_index])

        if len(coordinates) < 1000:
            for i in range(1000 - len(coordinates)):
                coordinates.append([float(0)] * 4)

    coordinates = coordinates[:1000]
    return coordinates