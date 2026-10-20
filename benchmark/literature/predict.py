"""
Our predictions for the compounds of the literature cases.

    uv run --extra ml python benchmark/literature/predict.py

Reads the tables in data/literature_benchmark/ (CASES) and predicts E_T, lambda_S0_T1 and
lambda_T1_S0 for every structure the model can encode (Ir and Ru complexes are skipped):
mixture mean and uncertainty of the deployed model (ml/data/model_final.pt).

Writes   results/literature/predictions_<case>.csv          one row per compound
"""

import sys

import numpy as np
import pandas as pd
from rdkit import Chem

from common import DATA, REPO, column, write

CASES = ("ma_2023", "sole_daura_2024", "leonori_2025", "bach_2026", "paton_2024")
HEADS = {"E_T": 0, "lambda_S0_T1": 1, "lambda_T1_S0": 2}   # order of the model's graph targets


def _imports():
    """The model code, imported without writing bytecode into the repository."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO))
    from ml.utils.graph_construction import ATOM_FEATURES
    from ml.prediction import collect_results

    return collect_results, ATOM_FEATURES


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


def frame(base, mean, sd):
    """Predictions as table columns, one row per predictable compound."""
    out = base.copy()
    for prop, i in HEADS.items():
        out[column(prop)] = mean[:, i]
        out[column(prop, uncertainty=True)] = sd[:, i]
    return out


def main():
    collect_results, atoms = _imports()
    elements = set(atoms["atomic_num"])
    tables = {case: compounds(pd.read_csv(DATA / f"{case}.csv"), elements) for case in CASES}
    smiles = sorted({s for t in tables.values() for s in t[t.predictable].smiles})
    index = {s: i for i, s in enumerate(smiles)}

    graph = collect_results(smiles)[0]
    mean, sd = np.asarray(graph["predictions"]), np.asarray(graph["uncertainties"])

    for case, table in tables.items():
        ok = table[table.predictable]
        rows = ok.smiles.map(index).to_numpy()
        f = frame(ok, mean[rows], sd[rows])
        write(pd.concat([f, table[~table.predictable]], ignore_index=True)
              .reindex(columns=list(table.columns) + [c for c in f.columns if c not in table.columns]),
              f"predictions_{case}.csv")
        skipped = table[~table.predictable]
        if len(skipped):
            print(f"  {case}: not predicted -- " + "; ".join(f"{r.compound_id} ({r.reason})" for r in skipped.itertuples()))


if __name__ == "__main__":
    main()
