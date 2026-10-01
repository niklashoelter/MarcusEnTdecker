import csv

import pytest
import torch

from ml.model import checkpoint as _checkpoint  # noqa: F401
from ml.training.hyperparameters import HYPERPARAMETERS, identify
from ml.training.splits import RANDOM_SEEDS
from ml.training.test_predictions import RESULTS, check_set, run_dir

OPTUNA = RESULTS / "hyperparameter_optimization" / "optuna_results.csv"


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


def test_trial_55_is_the_optuna_trial_exactly():
    assert HYPERPARAMETERS["trial_55"] == _trial(55)


def test_trial_55_has_the_lowest_validation_loss_of_the_search():
    with open(OPTUNA, newline="", encoding="utf-8") as f:
        complete = [r for r in csv.DictReader(f) if r["state"] == "COMPLETE"]
    assert min(complete, key=lambda r: float(r["value"]))["number"] == "55"


def test_trial_0_is_the_optuna_trial_as_leon_rounded_it():
    expected, ours = _trial(0), HYPERPARAMETERS["trial_0"]
    for key, value in expected.items():
        if isinstance(value, float):
            assert ours[key] == pytest.approx(value, rel=1e-2), key
        else:
            assert ours[key] == value, key


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_leons_checkpoints_are_trial_0(seed):
    path = run_dir("trial_0", f"random_{seed}") / "model.pt"
    config = torch.load(path, map_location="cpu", weights_only=False)["model_config"]
    assert identify(config) == "trial_0"
    with pytest.raises(ValueError):
        check_set("trial_55", path)


def test_the_two_sets_are_told_apart():
    assert identify(HYPERPARAMETERS["trial_55"]) == "trial_55"
    assert identify(HYPERPARAMETERS["trial_0"]) == "trial_0"
    assert identify({**HYPERPARAMETERS["trial_55"], "walk_length": 6}) is None
