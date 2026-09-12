"""The checkpoint predates the package rename; make sure it still unpickles."""

import importlib.resources as ir
from pathlib import Path

import torch

from ml import _checkpoint_compat  # noqa: F401


def test_checkpoint_scaler_resolves_to_the_new_module():
    model_path = Path(str(ir.files("ml.data").joinpath("model.pt")))
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    assert type(ckpt["scaler"]).__module__ == "ml.utilities.dataset.molecule_dataset"
