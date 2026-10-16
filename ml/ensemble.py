"""
The five random-split models as one ensemble.

    from ml.ensemble import predict
    predict(["c1ccccc1"])     # the columns of ml.prediction.predict, plus *_model_spread

    from ml.predict import RatePredictor
    RatePredictor(ensemble=True).predict(photocatalyst, quencher)

    ent-ml predict --ensemble SMILES ...
    ent-ml rate --ensemble --quencher SMILES --photocatalysts TABLE

Written for this repository, not taken from Leon Schlosser's code. Members:
ml/results/random_<seed>/model.pt, listed with md5 sums in ml/results/ensemble_members.csv;
a member whose md5 does not match is refused. Not in the installed package; outside a
checkout, pass `checkpoints=`.

Equal-weight mixture of the M members' Gaussian mixtures (weights / M), with member mean m_i
and SD s_i:

    mean = (1/M) sum_i m_i
    var  = (1/M) sum_i s_i^2  +  (1/M) sum_i (m_i - mean)^2

`<target>_model_spread` is the square root of the second term.
"""

from __future__ import annotations

import hashlib
import importlib.resources as ir
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

RESULTS = Path(str(ir.files("ml"))) / "results"
MEMBERS = RESULTS / "ensemble_members.csv"
BATCH_SIZE = 48

#: Column names of the graph targets in the prediction table, in model output order.
GRAPH_COLUMNS = ("triplet_energy_adiabatic", "lamda_S_T", "lamda_T_S")


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def default_checkpoints() -> list[Path]:
    """Checkpoints listed in ml/results/ensemble_members.csv, md5-checked."""
    import csv

    if not MEMBERS.exists():
        raise FileNotFoundError(f"{MEMBERS} does not exist; ml.training.evaluate writes it")
    root = RESULTS.parent.parent
    with open(MEMBERS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    paths = []
    for row in rows:
        path = root / row["checkpoint"]
        if md5(path) != row["md5"]:
            raise ValueError(f"{path} is not the ensemble member {MEMBERS.name} records")
        paths.append(path)
    return paths


def combine(outputs: list[dict]) -> dict:
    """One prediction from the members' outputs, each as ml.model.inference.predict returns it."""
    combined = {}
    n = len(outputs)
    for head in ("graph", "node"):
        heads = [output[head] for output in outputs]
        means = np.stack([h["predictions"] for h in heads])
        sds = np.stack([h["uncertainties"] for h in heads])
        mean = means.mean(axis=0)
        spread = means.std(axis=0)
        mdn = {k: np.concatenate([h["mdn_params"][k] for h in heads], axis=-1)
               for k in ("mu", "sigma", "pi")}
        mdn["pi"] = mdn["pi"] / n
        combined[head] = {"predictions": mean,
                          "uncertainties": np.sqrt((sds ** 2).mean(axis=0) + spread ** 2),
                          "model_spread": spread,
                          "weights": mdn["pi"],
                          "targets": None,
                          "mdn_params": mdn}
    return combined


class Ensemble:
    """Several checkpoints that predict together. The models load on the first prediction."""

    def __init__(self, checkpoints: Iterable[str | Path] | None = None, device: str = "cpu"):
        self.checkpoints = ([Path(p) for p in checkpoints] if checkpoints is not None
                            else default_checkpoints())
        if not self.checkpoints:
            raise ValueError("an ensemble needs at least one checkpoint")
        self.device = device
        self._members = None
        self._lock = threading.Lock()

    def _load(self, smiles: list[str]) -> list[tuple]:
        import torch

        from ml.model import checkpoint as _checkpoint  # noqa: F401  aliases the pre-rename path
        from ml.model.inference import build_model

        members = []
        for path in self.checkpoints:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            model = build_model(checkpoint, self.device, smiles, batch_size=BATCH_SIZE)
            members.append((model, checkpoint["scaler"], checkpoint["model_config"]))
        return members

    def outputs(self, smiles: Iterable[str]) -> dict:
        """The combined graph and node outputs, shaped as ml.model.inference.predict's."""
        from torch.utils.data import DataLoader

        from ml.model.inference import predict
        from ml.utils.molecule_dataset import MoleculeDataset, collate_fn

        smiles = list(smiles)
        with self._lock:
            if self._members is None:
                self._members = self._load(smiles)
            results = []
            for model, scaler, config in self._members:
                dataset = MoleculeDataset(smiles, positional_encoding_kwargs=config)
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                                    collate_fn=collate_fn)
                results.append(predict(model, loader, self.device, scaler))
        return combine(results)

    def predict(self, smiles: Iterable[str]):
        """The table of ml.prediction.predict, plus the spread between the members."""
        from rdkit import Chem

        from ml.prediction import frame

        smiles = list(smiles)
        out = self.outputs(smiles)
        table = frame(smiles, out["graph"], out["node"])

        for i, name in enumerate(GRAPH_COLUMNS):
            table[f"{name}_model_spread"] = out["graph"]["model_spread"][:, i]
        atoms = [Chem.MolFromSmiles(s).GetNumAtoms() for s in smiles]
        per_molecule = np.split(out["node"]["model_spread"].ravel(), np.cumsum(atoms)[:-1])
        table["spin_map_model_spread"] = [part.tolist() for part in per_molecule]
        table["n_models"] = len(self.checkpoints)
        return table


_default: Ensemble | None = None
_default_lock = threading.Lock()


def predict(smiles: Iterable[str], checkpoints: Iterable[str | Path] | None = None,
            device: str = "cpu"):
    """Predict with the ensemble; the default (random-split) ensemble is cached."""
    global _default
    if checkpoints is not None:
        return Ensemble(checkpoints, device).predict(smiles)
    with _default_lock:
        if _default is None:
            _default = Ensemble(device=device)
    return _default.predict(smiles)
