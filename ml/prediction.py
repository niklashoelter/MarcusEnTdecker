from __future__ import annotations
from pathlib import Path
from typing import Iterable
from ml.model.inference import get_predictions
import importlib.resources as ir
import pandas as pd
import numpy as np
import threading

_model_lock = threading.Lock()

def collect_results(smiles: list):
    """Collect prediction values"""

    with _model_lock:
        output = get_predictions(smiles, model_path=Path(str(ir.files("ml.data").joinpath("model_final.pt"))), device='cpu', task='all')
    graph_output = output['graph']
    node_output = output['node']

    return graph_output, node_output

def predict(smiles_list: Iterable[str]) -> pd.DataFrame:
    """Stub prediction function to be replaced by real ML models.

    Returns a DataFrame with columns (base unit: kcal/mol):
      - smiles
      - triplet_energy_adiabatic        (in kcal/mol, constrained to [30, 75])
      - lamda_S_T                        (in kcal/mol)
      - lamda_T_S                        (in kcal/mol)
      - spin_map                         (dict: atom_idx -> spin density) [internal usage, hidden in table]
    """
    graph_output, node_output = collect_results(smiles_list)
    return frame(smiles_list, graph_output, node_output)


def frame(smiles_list, graph_output, node_output) -> pd.DataFrame:
    """The prediction table of `predict`, from the graph and node outputs of a model."""
    from rdkit import Chem

    graph_predictions, graph_uncertainties = graph_output['predictions'], graph_output['uncertainties']
    g_mu, g_sigma, g_pi = graph_output['mdn_params']['mu'], graph_output['mdn_params']['sigma'], graph_output['mdn_params']['pi']
    node_predictions_flat, node_uncertainties_flat = node_output['predictions'], node_output['uncertainties']
    n_mu_flat, n_sigma_flat, n_pi_flat = node_output['mdn_params']['mu'], node_output['mdn_params']['sigma'], node_output['mdn_params']['pi']


    # Calculate number of atoms per molecule
    num_atoms_list = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
             num_atoms_list.append(mol.GetNumAtoms())
        else:
             num_atoms_list.append(0)

    # Split flat node predictions by molecule
    node_predictions = []
    node_uncertainties = []
    start_idx = 0
    for num_atoms in num_atoms_list:
        end_idx = start_idx + num_atoms
        node_predictions.append(node_predictions_flat[start_idx:end_idx])
        node_uncertainties.append(node_uncertainties_flat[start_idx:end_idx])
        start_idx = end_idx

    rows = []
    for i, smiles in enumerate(smiles_list):
        rows.append(
            {
                "smiles": smiles,
                "triplet_energy_adiabatic": graph_predictions[i,0],
                "triplet_energy_adiabatic_uncertainty": graph_uncertainties[i,0],
                "triplet_energy_adiabatic_mdn_mu": g_mu[i,0,:].tolist(),
                "triplet_energy_adiabatic_mdn_sigma": g_sigma[i,0,:].tolist(),
                "triplet_energy_adiabatic_mdn_pi": g_pi[i,0,:].tolist(),
                "lamda_S_T": graph_predictions[i,1],
                "lamda_S_T_uncertainty": graph_uncertainties[i,1],
                "lamda_S_T_mdn_mu": g_mu[i,1,:].tolist(),
                "lamda_S_T_mdn_sigma": g_sigma[i,1,:].tolist(),
                "lamda_S_T_mdn_pi": g_pi[i,1,:].tolist(),
                "lamda_T_S": graph_predictions[i,2],
                "lamda_T_S_uncertainty": graph_uncertainties[i,2],
                "lamda_T_S_mdn_mu": g_mu[i,2,:].tolist(),
                "lamda_T_S_mdn_sigma": g_sigma[i,2,:].tolist(),
                "lamda_T_S_mdn_pi": g_pi[i,2,:].tolist(),
                "spin_map": np.array(node_predictions[i]).flatten().tolist(),
                "spin_map_uncertainty": np.array(node_uncertainties[i]).flatten().tolist(),
            }
        )

    df = pd.DataFrame(rows)

    return df
