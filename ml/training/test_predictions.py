"""
Test-set predictions of a trained checkpoint.

Reads   ml/results/<set>/<split>/model.pt              the checkpoint trained on that split
        ml/data/splits/<split>.csv, ml/data/training_dataset_full.csv
Writes  ml/results/<set>/<split>/test_predictions_graph.csv   one row per test molecule
        ml/results/<set>/<split>/test_predictions_node.csv    one row per atom of a test molecule

    uv run --extra ml python -m ml.training.test_predictions trial_0 random_42 random_101 ...

<set> is a name from hyperparameters.py. The checkpoint must have been trained with that set:
its stored model configuration is checked against the set before anything is written.

`train.py` writes the same two files at the end of a run; this module also serves
checkpoints that were trained elsewhere. Adapted from Leon Schlosser's
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

from ml.training.hyperparameters import HYPERPARAMETERS, identify
from ml.training.splits import DATASET, read_split

RESULTS = Path(str(ir.files("ml"))) / "results"

GRAPH_TARGETS = ["adiabatic_energy", "lambda_acceptor_singlet", "lambda_donor_triplet"]
NODE_TARGETS = ["spin_population"]
ALL_TARGETS = GRAPH_TARGETS + NODE_TARGETS


def run_dir(name, split):
    """ml/results/<set>/<split>, for a known set only."""
    if name not in HYPERPARAMETERS:
        raise ValueError(f"unknown hyperparameter set: {name}")
    return RESULTS / name / split


def check_set(name, checkpoint_path):
    """Refuse a checkpoint whose model configuration is not that of set `name`."""
    import torch

    from ml.model import checkpoint as _checkpoint  # noqa: F401  aliases the pre-rename path

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    found = identify(checkpoint["model_config"])
    if found != name:
        raise ValueError(f"{checkpoint_path} was trained with {found}, not {name}")
    if checkpoint.get("hyperparameters", name) != name:
        raise ValueError(f"{checkpoint_path} is labelled {checkpoint['hyperparameters']}, not {name}")
    return checkpoint


def load_dataset():
    """The training dataset, with spin populations parsed as Leon's `compute_targets` does."""
    data = pd.read_csv(DATASET)
    data["spin_population"] = data["spin_population"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return data


def write_test_predictions(name, split, device="cpu", data=None):
    """Predict the test molecules of `split` with set `name`'s checkpoint; write both CSVs."""
    from ml.model import inference

    out_dir = run_dir(name, split)
    checkpoint_path = out_dir / "model.pt"
    check_set(name, checkpoint_path)

    data = load_dataset() if data is None else data
    _, _, test_idx = read_split(split)
    test = data.iloc[test_idx].reset_index(drop=True)

    inference._global_model_cache = None  # get_predictions keeps the first model it loads
    result = inference.get_predictions(test["smiles"].tolist(), model_path=checkpoint_path,
                                       device=device, task="all")
    inference._global_model_cache = None

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
    parser.add_argument("hyperparameters", choices=sorted(HYPERPARAMETERS))
    parser.add_argument("splits", nargs="+", help="split names, e.g. random_42 oos_styryl")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data = load_dataset()
    for split in args.splits:
        graph, node = write_test_predictions(args.hyperparameters, split, args.device, data)
        print(f"{args.hyperparameters} {split}: {len(graph)} molecules, {len(node)} atoms")


if __name__ == "__main__":
    main()
