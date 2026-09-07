import torch
from torch_geometric.data import Data
from rdkit import Chem

# Define allowed values for one-hot encoding
ATOM_FEATURES = {
    'atomic_num': [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 50, 53],  # H, B, C, N, O, F, Si, P, S, Cl, Se, Br, Sn, I
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    'num_hs': [0, 1, 2, 3, 4],
    'chiral_tag': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ],
}

BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
        Chem.rdchem.BondStereo.STEREOANY,
    ],
}


class UnknownAtomError(ValueError):
    """Raised when an atom cannot be encoded."""
    pass


def one_hot_encode(value, allowed_values, feature_name="feature"):
    """One-hot encode a value. Raises an error for unknown atom types."""
    if feature_name == "atomic_num":
        encoding = [0] * len(allowed_values)
        if value in allowed_values:
            encoding[allowed_values.index(value)] = 1
        else:
            raise UnknownAtomError(
                f"Cannot encode atom with atomic number {value}. "
                f"Allowed atomic numbers are: {allowed_values}"
            )
        return encoding

    # For other features, always include an extra position for "unknown" values
    encoding = [0] * (len(allowed_values) + 1)
    if value in allowed_values:
        encoding[allowed_values.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding


def get_atom_features(atom):
    """Extract and one-hot encode atom features."""
    features = []
    features += one_hot_encode(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'], "atomic_num")
    features += one_hot_encode(atom.GetDegree(), ATOM_FEATURES['degree'], "degree")
    features += one_hot_encode(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'], "formal_charge")
    features += one_hot_encode(atom.GetHybridization(), ATOM_FEATURES['hybridization'], "hybridization")
    features += one_hot_encode(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs'], "num_hs")
    features += one_hot_encode(atom.GetChiralTag(), ATOM_FEATURES['chiral_tag'], "chiral_tag")
    features += [float(atom.GetIsAromatic())]
    return features


def get_bond_features(bond):
    """Extract and one-hot encode bond features."""
    features = []
    features += one_hot_encode(bond.GetBondType(), BOND_FEATURES['bond_type'], "bond_type")
    features += one_hot_encode(bond.GetStereo(), BOND_FEATURES['stereo'], "stereo")
    features += [float(bond.GetIsConjugated())]
    features += [float(bond.IsInRing())]
    return features


def smiles_to_graph(smiles, use_explicit_hydrogens=False, use_stereochemistry=True):
    """
    Converts a SMILES string to a graph representation suitable for a directed MPNN.

    The graph includes an index `rev_edge_index` that maps each directed edge
    to its reverse edge, which is crucial for the message passing phase in D-MPNNs.

    Includes one-hot encoded atom and bond features including stereochemistry.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Enable stereochemistry perception
    if use_stereochemistry:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # Add explicit hydrogens
    if use_explicit_hydrogens:
        mol = Chem.AddHs(mol)

    # Atom features
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(get_atom_features(atom))
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Edge features and indices
    edge_indices = []
    edge_features = []
    if mol.GetNumBonds() > 0:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bond_feature = get_bond_features(bond)

            # Add both directed edges
            edge_indices.append([i, j])
            edge_features.append(bond_feature)
            edge_indices.append([j, i])
            edge_features.append(bond_feature)

    # Calculate feature dimensions for empty graphs
    # Bond type (4+1) + stereo (6+1) + conjugated (1) + in_ring (1) = 14
    edge_attr_dim = len(BOND_FEATURES['bond_type']) + 1 + len(BOND_FEATURES['stereo']) + 1 + 2

    # Create graph attributes
    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, edge_attr_dim), dtype=torch.float)
        rev_edge_index = torch.zeros((0,), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # Create reverse edge index
        num_edges = edge_index.size(1)
        rev_edge_index = torch.zeros(num_edges, dtype=torch.long)
        for i in range(num_edges):
            rev_edge_index[i] = i + 1 if i % 2 == 0 else i - 1

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, rev_edge_index=rev_edge_index, smiles=smiles)

