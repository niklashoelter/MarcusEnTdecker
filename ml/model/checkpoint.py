"""Let the checkpoint unpickle after the rename.

model_final_old.pt, Leon Schlosser's checkpoint, was saved when this package was called
`pyrates`, and torch.load resolves the pickled `DualScaler` by its dotted path. Aliasing the
old path onto the new module keeps that file byte-identical instead of re-saving it. The
checkpoints trained in this repository need no alias.
"""

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
