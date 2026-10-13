"""
Our predictions for the compounds of the literature cases.

    uv run --extra ml python benchmark/literature/predict.py

Reads the tables in data/literature_benchmark/ (CASES) and predicts E_T and both
reorganisation energies for every structure the model can encode. The comparison scripts
beside this one read what it writes; none of them calls the model.

WHAT IS PREDICTED
    Every substrate, and every photocatalyst that is an ordinary organic molecule: for example
    Leonori's nitroarenes and the thioxanthones. Iridium and ruthenium complexes cannot be
    encoded -- their elements are not in the model's atom vocabulary -- so they are skipped
    here and their parameters come from the publications.

TWO ESTIMATES, SIX MODELS
    The deployed model (ml/data/model_final.pt) predicts a mixture of Gaussians per property.
    Both its mean and its highest-weight component (`ml.model.ffn.mdn_utils.compute_max_mode`)
    are written. The five random-split members of `ml.ensemble` are then predicted one at a
    time with the same two estimates.

Writes   results/literature/predictions_<case>.csv          deployed model, one row per compound
         results/literature/predictions_members_<case>.csv  deployed model and every member
"""

import sys

import numpy as np
import pandas as pd
from rdkit import Chem

from common import DATA, PROPERTIES, REPO, column, write

CASES = ("ma_2023", "sole_daura_2024", "leonori_2025", "gilmour_2020", "bach_2026", "booker_milburn_2020",
         "paton_2024")
HEADS = {"E_T": 0, "lambda_S0_T1": 1, "lambda_T1_S0": 2}   # order of the model's graph targets


def _imports():
    """The model code, imported without writing bytecode into the repository."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO))
    import torch
    from ml.ensemble import Ensemble, default_checkpoints
    from ml.utils.graph_construction import ATOM_FEATURES
    from ml.model.ffn.mdn_utils import compute_max_mode
    from ml.prediction import collect_results

    return torch, Ensemble, default_checkpoints, compute_max_mode, collect_results, ATOM_FEATURES


def encodable(smiles, elements):
    """Whether the model can encode a structure, and why not."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return False, "no parsable SMILES"
    heavy = sorted({a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() not in elements})
    if heavy:
        return False, f"element(s) {', '.join(heavy)} outside the model vocabulary"
    return True, ""


def compounds(table, elements):
    """One row per distinct structure: the substrates, and the photocatalysts of the table."""
    rows = []
    for _, mol in table.drop_duplicates(subset="compound_id").iterrows():
        rows.append({"compound_id": mol.compound_id, "label": mol.label, "name": mol.get("name"),
                     "smiles": mol.smiles, "role": mol.get("role", "substrate"),
                     "isomer": mol.get("isomer", "")})
    if "pc_smiles" in table:
        # catalyst-free control rows carry no catalyst SMILES and are not catalysts
        for _, pc in table.dropna(subset=["pc_smiles"]).drop_duplicates(subset="pc_label").iterrows():
            rows.append({"compound_id": f"PC_{pc.pc_label}", "label": pc.pc_label, "name": pc.pc_label,
                         "smiles": pc.pc_smiles, "role": "photocatalyst", "isomer": ""})
    out = pd.DataFrame(rows).drop_duplicates(subset="compound_id")
    checks = out.smiles.map(lambda smi: encodable(smi, elements))
    out["predictable"] = [c[0] for c in checks]
    out["reason"] = [c[1] for c in checks]
    return out


def estimates(graph, compute_max_mode, torch):
    """Mean, highest-weight component and uncertainty per molecule and property."""
    mdn = graph["mdn_params"]
    mode, _ = compute_max_mode(torch.as_tensor(np.asarray(mdn["mu"])), torch.as_tensor(np.asarray(mdn["sigma"])),
                               torch.as_tensor(np.asarray(mdn["pi"])))
    return np.asarray(graph["predictions"]), mode.numpy(), np.asarray(graph["uncertainties"])


def frame(base, mean, mode, sd):
    """Predictions as table columns, one row per predictable compound."""
    out = base.copy()
    for prop, i in HEADS.items():
        out[column(prop)] = mean[:, i]
        out[column(prop, "mode")] = mode[:, i]
        out[column(prop, uncertainty=True)] = sd[:, i]
    return out


def main():
    torch, Ensemble, default_checkpoints, compute_max_mode, collect_results, atoms = _imports()
    elements = set(atoms["atomic_num"])
    tables = {case: compounds(pd.read_csv(DATA / f"{case}.csv"), elements) for case in CASES}
    smiles = sorted({s for t in tables.values() for s in t[t.predictable].smiles})
    index = {s: i for i, s in enumerate(smiles)}

    runs = {"deployed": estimates(collect_results(smiles)[0], compute_max_mode, torch)}
    for checkpoint in default_checkpoints():
        runs[checkpoint.parent.name] = estimates(Ensemble([checkpoint]).outputs(smiles)["graph"],
                                                 compute_max_mode, torch)

    for case, table in tables.items():
        ok = table[table.predictable]
        rows = [ok.smiles.map(index).to_numpy()]
        members = []
        for name, (mean, mode, sd) in runs.items():
            f = frame(ok, mean[rows[0]], mode[rows[0]], sd[rows[0]])
            if name == "deployed":
                write(pd.concat([f, table[~table.predictable]], ignore_index=True)
                      .reindex(columns=list(table.columns) + [c for c in f.columns if c not in table.columns]),
                      f"predictions_{case}.csv")
            members.append(f.assign(model=name))
        write(pd.concat(members, ignore_index=True), f"predictions_members_{case}.csv")
        skipped = table[~table.predictable]
        if len(skipped):
            print(f"  {case}: not predicted -- " + "; ".join(f"{r.compound_id} ({r.reason})" for r in skipped.itertuples()))


if __name__ == "__main__":
    main()
