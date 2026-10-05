"""
Test-set predictions of a trained checkpoint.

Reads   ml/results/<split>/model.pt              the checkpoint trained on that split
        ml/data/splits/<split>.csv, ml/data/training_dataset_full.csv
Writes  ml/results/<split>/test_predictions_graph.csv   one row per test molecule
        ml/results/<split>/test_predictions_node.csv    one row per atom of a test molecule

    uv run --extra ml python -m ml.training.test_predictions random_42 random_101 ...

`train.py` writes the same two files at the end of a run. Adapted from Leon Schlosser's
playground/thesis/model/gps/get_preds.py and the output block of
playground/model_setup/save_model_and_inspect/new_data/graph_gps/og_data_opt_param/train.py:
the columns are those of the training script (`<target>_target`, `_pred`, `_uncertainty`).
One change: his per-atom file labelled only its first rows with a molecule `idx`, so each
atom row here carries its molecule's `idx` and its position in the molecule. The values
are unaffected.
"""

import argparse
import ast
import importlib.resources as ir
from pathlib import Path

import numpy as np
import pandas as pd

from ml.training.splits import DATASET, read_split

RESULTS = Path(str(ir.files("ml"))) / "results"

GRAPH_TARGETS = ["adiabatic_energy", "lambda_acceptor_singlet", "lambda_donor_triplet"]
NODE_TARGETS = ["spin_population"]
ALL_TARGETS = GRAPH_TARGETS + NODE_TARGETS


def run_dir(split):
    """ml/results/<split>."""
    return RESULTS / split


def load_dataset():
    """The training dataset, with spin populations parsed as Leon's `compute_targets` does."""
    data = pd.read_csv(DATASET)
    data["spin_population"] = data["spin_population"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return data


def write_test_predictions(split, device="cpu", data=None):
    """Predict the test molecules of `split` with its checkpoint; write both CSVs."""
    from ml.model import inference

    out_dir = run_dir(split)
    checkpoint_path = out_dir / "model.pt"

    data = load_dataset() if data is None else data
    _, _, test_idx = read_split(split)
    test = data.iloc[test_idx].reset_index(drop=True)

    result = inference.get_predictions(test["smiles"].tolist(), model_path=checkpoint_path,
                                       device=device, task="all")

    graph = pd.DataFrame({"idx": test["idx"], "smiles": test["smiles"]})
    for i, target in enumerate(GRAPH_TARGETS):
        graph[f"{target}_target"] = test[target].to_numpy()
        graph[f"{target}_pred"] = result["graph"]["predictions"][:, i]
        graph[f"{target}_uncertainty"] = result["graph"]["uncertainties"][:, i]

    atoms = test["spin_population"].map(len).to_numpy()
    node = pd.DataFrame({"idx": np.repeat(test["idx"].to_numpy(), atoms),
                         "atom": np.concatenate([np.arange(k) for k in atoms])})
    node["spin_population_target"] = np.concatenate(test["spin_population"].tolist())
    node_pred = result["node"]["predictions"].ravel()
    if len(node_pred) != len(node):
        raise ValueError(f"{split}: {len(node_pred)} atom predictions for {len(node)} atoms")
    node["spin_population_pred"] = node_pred
    node["spin_population_uncertainty"] = result["node"]["uncertainties"].ravel()

    graph.to_csv(out_dir / "test_predictions_graph.csv", index=False, lineterminator="\n")
    node.to_csv(out_dir / "test_predictions_node.csv", index=False, lineterminator="\n")
    return graph, node


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("splits", nargs="+", help="split names, e.g. random_42 oos_styryl")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data = load_dataset()
    for split in args.splits:
        graph, node = write_test_predictions(split, args.device, data)
        print(f"{split}: {len(graph)} molecules, {len(node)} atoms")


if __name__ == "__main__":
    main()
