"""Alias the `pyrates` module paths so the pickled `DualScaler` in model_final_old.pt
(Leon Schlosser's checkpoint) resolves on torch.load."""

import sys
import types

import ml.utils.molecule_dataset as _molecule_dataset

for _name, _module in (
    ("pyrates", types.ModuleType("pyrates")),
    ("pyrates.utilities", types.ModuleType("pyrates.utilities")),
    ("pyrates.utilities.dataset", types.ModuleType("pyrates.utilities.dataset")),
    ("pyrates.utilities.dataset.molecule_dataset", _molecule_dataset),
):
    sys.modules.setdefault(_name, _module)
