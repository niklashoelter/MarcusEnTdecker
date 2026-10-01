"""Let the checkpoint unpickle after the rename.

model_final.pt was saved when this package was called `pyrates`, and torch.load resolves
the pickled `DualScaler` by its dotted path. Aliasing the old path onto the new module
keeps the 69 MB artifact the paper pins byte-identical instead of re-saving it.
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
