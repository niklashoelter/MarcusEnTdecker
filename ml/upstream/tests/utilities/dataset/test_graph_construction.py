import torch
import pytest
from rdkit import Chem
from pyrates.utilities.dataset.graph_construction import smiles_to_graph, UnknownAtomError

def test_smiles_to_graph_methane():
    smiles = "C"
    data = smiles_to_graph(smiles)

    # Methane structure: C with 4 H. Implicit H are usually part of node features or implicit.
    # rdkit MolFromSmiles("C") gives 1 atom if implicit H.
    # Let's check node features.

    # one_hot_encode features length:
    # atomic_num: 14 values
    # degree: 6
    # formal_charge: 5
    # hybridization: 5
    # num_hs: 5
    # chiral_tag: 3
    # aromatic: 1
    # mass: 1
    # total: 14+6+5+5+5+3+1+1 = 40 (approx, need to check exact implementation)

    assert data.x.shape[0] == 1 # 1 Carbon atom
    assert data.edge_index.shape[1] == 0 # No bonds between heavy atoms

def test_smiles_to_graph_ethane():
    smiles = "CC"
    data = smiles_to_graph(smiles)

    assert data.x.shape[0] == 2 # 2 Carbon atoms
    # 1 bond forward, 1 backward = 2 edges
    assert data.edge_index.shape[1] == 2

def test_features_consistency():
    smiles = "C"
    data = smiles_to_graph(smiles)
    # Check if features are float tensors
    assert data.x.dtype == torch.float
    if data.edge_attr is not None:
        assert data.edge_attr.dtype == torch.float

def test_rev_edge_index_generation():
    # Test for simple molecule C-C
    smiles = "CC"
    data = smiles_to_graph(smiles)

    # Expect 2 edges: (0,1) and (1,0)
    # edge_index should be [[0, 1], [1, 0]]
    # rev_edge_index should be [1, 0]

    assert data.edge_index.shape[1] == 2

    # Check edge connectivity
    u, v = data.edge_index
    assert u[0] == 0 and v[0] == 1
    assert u[1] == 1 and v[1] == 0

    # Check rev_edge_index
    assert hasattr(data, 'rev_edge_index')
    rev = data.rev_edge_index
    assert rev[0] == 1 # 0->1 reverse is 1->0 (index 1)
    assert rev[1] == 0 # 1->0 reverse is 0->1 (index 0)

def test_rev_edge_index_ring():
    # Test Cyclopropane C1CC1
    smiles = "C1CC1"
    data = smiles_to_graph(smiles)
    # 3 atoms. 3 bonds. 6 directed edges.
    # Indices: 0, 1, 2, 3, 4, 5
    # Pairs: (0,1), (2,3), (4,5) are reverses.

    assert data.edge_index.shape[1] == 6
    rev = data.rev_edge_index
    assert len(rev) == 6

    for i in range(0, 6, 2):
        assert rev[i] == i + 1
        assert rev[i+1] == i
