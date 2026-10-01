"""The checkpoint predates the package rename; make sure it still unpickles."""

import importlib.resources as ir
from pathlib import Path

import torch

from ml.model import checkpoint as _checkpoint  # noqa: F401


def test_checkpoint_scaler_resolves_to_the_new_module():
    model_path = Path(str(ir.files("ml.data").joinpath("model_final.pt")))
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    assert type(ckpt["scaler"]).__module__ == "ml.utils.molecule_dataset"
