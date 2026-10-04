import csv
import importlib.resources as ir
from pathlib import Path

import pytest
import torch

from ml.training.hyperparameters import HYPERPARAMETERS, NAME
from ml.training.splits import RANDOM_SEEDS
from ml.training.test_predictions import RESULTS, run_dir

OPTUNA = RESULTS / "hyperparameter_optimization" / "optuna_results.csv"
MODEL_FINAL = Path(str(ir.files("ml.data").joinpath("model_final.pt")))
ARCHITECTURE_KEYS = ("node_hidden_dim", "encoder_layers", "encoder_dropout", "num_heads",
                     "walk_length", "pe_dim", "n_components", "mdn_hidden_dims", "mdn_dropout")


def _trial(number):
    with open(OPTUNA, newline="", encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["number"] == str(number))
    p = {k[len("params_"):]: v for k, v in row.items() if k.startswith("params_")}
    return {
        "node_hidden_dim": int(p["base_node_hidden_dim"]) * int(p["num_heads"]),
        "encoder_layers": int(p["encoder_layers"]),
        "encoder_dropout": float(p["encoder_dropout"]),
        "num_heads": int(p["num_heads"]),
        "walk_length": int(p["walk_length"]),
        "pe_dim": int(p["pe_dim"]),
        "n_components": int(p["n_components"]),
        "mdn_hidden_dims": [int(p["mdn_hidden_dim_size"])] * int(p["n_mdn_layers"]),
        "mdn_dropout": float(p["mdn_dropout"]),
        "lr": float(p["lr"]),
        "weight_decay": float(p["weight_decay"]),
    }


def test_the_hyperparameters_are_the_best_optuna_trial_exactly():
    with open(OPTUNA, newline="", encoding="utf-8") as f:
        complete = [r for r in csv.DictReader(f) if r["state"] == "COMPLETE"]
    best = min(complete, key=lambda r: float(r["value"]))["number"]
    assert NAME == f"trial_{best}"
    assert HYPERPARAMETERS == _trial(int(best))


@pytest.mark.parametrize("split", [f"random_{s}" for s in RANDOM_SEEDS] + ["oos_styryl", "full"])
def test_the_committed_checkpoints_were_trained_with_them(split):
    checkpoint = torch.load(run_dir(split) / "model.pt", map_location="cpu", weights_only=False)
    assert checkpoint["hyperparameters"] == NAME and checkpoint["split"] == split
    for key in ARCHITECTURE_KEYS:
        assert checkpoint["model_config"][key] == HYPERPARAMETERS[key], key


def test_the_deployed_model_is_the_full_data_model():
    assert MODEL_FINAL.read_bytes() == (run_dir("full") / "model.pt").read_bytes()
