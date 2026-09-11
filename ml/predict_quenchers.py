"""
Predict triplet energies and reorganization energies for the 48 screened quenchers.

Reads   ../data/experimental_data_raw/compound_records/compound_records_quenchers.csv   (SMILES)
        ../data/computational_data_raw/quencher_adiabatic_DFT/quencher_dft.csv          (comparison)

The same values are in the property workbooks in `data/*_processed/`. The intermediate
CSVs are read instead so that the model environment does not need a spreadsheet reader.
Writes  predictions_quenchers.csv

Run it with the model environment, which is not the one the data pipeline uses:

    uv sync --extra ml
    uv run --extra ml python ml/predict_quenchers.py

This is the one place where the model meets compounds whose DFT values are already
known from first principles, so it doubles as the check on whether the checkpoint is
loading and running correctly. The comparison columns are written for that purpose,
not because a prediction should ever be replaced by the DFT number -- they are separate
quantities with separate provenance, suffixed `_ml` and `_dft`.

WHY THE QUENCHERS AND NOT THE PAIRS
    A rate constant needs both partners, and the photocatalyst-side DFT is not in the
    repository yet. Predicting the quenchers alone is worth doing on its own: it is the
    half of the problem the model can be checked on today, against 48 compounds whose
    adiabatic triplet energies were computed independently.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDS = (ROOT / "data" / "experimental_data_raw" / "compound_records"
           / "compound_records_quenchers.csv")
DFT = ROOT / "data" / "computational_data_raw" / "quencher_adiabatic_DFT" / "quencher_dft.csv"
OUT = Path(__file__).resolve().parent / "predictions_quenchers.csv"

SIGFIGS = 6

COLUMNS = [
    "quencher_id", "quencher_name", "name", "smiles",
    "E_T_ml / kcal mol-1", "E_T_ml_uncertainty / kcal mol-1",
    "lambda_S_T_ml / kcal mol-1", "lambda_S_T_ml_uncertainty / kcal mol-1",
    "lambda_T_S_ml / kcal mol-1", "lambda_T_S_ml_uncertainty / kcal mol-1",
    "E_T_dft / kcal mol-1", "E_T_error / kcal mol-1",
    "lambda_S_T_dft / kcal mol-1", "lambda_T_S_dft / kcal mol-1",
]


def sig(x, n=SIGFIGS):
    if x is None:
        return ""
    return 0.0 if float(x) == 0 else float(f"{float(x):.{n}g}")


def read_csv(path, key):
    if not path.exists():
        sys.exit(f"missing input: {path}\nrun the data stage first: uv run reproduce.py data")
    with path.open(encoding="utf-8", newline="") as fh:
        return {int(row[key]): row for row in csv.DictReader(fh)}


def number(row, field):
    if row is None:
        return None
    value = row.get(field, "")
    return float(value) if value not in ("", None) else None


def main():
    records = read_csv(RECORDS, "quencher_id")
    dft = read_csv(DFT, "quencher_id")

    ordered = sorted(records)
    smiles = [records[q]["smiles"] for q in ordered]
    missing = [q for q in ordered if not records[q]["smiles"]]
    if missing:
        sys.exit(f"quenchers without SMILES, cannot predict: {missing}")

    print(f"predicting {len(smiles)} quenchers ...", flush=True)
    from pyrates.webapp.science_utils.prediction import predict

    frame = predict(smiles)
    by_smiles = {row["smiles"]: row for _, row in frame.iterrows()}

    rows, errors = [], []
    for q in ordered:
        record = records[q]
        prediction = by_smiles.get(record["smiles"])
        if prediction is None:
            print(f"  Q{q}: no prediction returned")
            continue

        e_t_ml = float(prediction["triplet_energy_adiabatic"])
        e_t_dft = number(dft.get(q), "E_T")
        error = (e_t_ml - e_t_dft) if e_t_dft is not None else None
        if error is not None:
            errors.append(error)

        rows.append({
            "quencher_id": q,
            "quencher_name": record["quencher_name"],
            "name": record["name"],
            "smiles": record["smiles"],
            "E_T_ml / kcal mol-1": sig(e_t_ml),
            "E_T_ml_uncertainty / kcal mol-1":
                sig(prediction["triplet_energy_adiabatic_uncertainty"]),
            "lambda_S_T_ml / kcal mol-1": sig(prediction["lamda_S_T"]),
            "lambda_S_T_ml_uncertainty / kcal mol-1": sig(prediction["lamda_S_T_uncertainty"]),
            "lambda_T_S_ml / kcal mol-1": sig(prediction["lamda_T_S"]),
            "lambda_T_S_ml_uncertainty / kcal mol-1": sig(prediction["lamda_T_S_uncertainty"]),
            "E_T_dft / kcal mol-1": sig(e_t_dft),
            "E_T_error / kcal mol-1": sig(error),
            "lambda_S_T_dft / kcal mol-1": sig(number(dft.get(q), "lambda_S_T")),
            "lambda_T_S_dft / kcal mol-1": sig(number(dft.get(q), "lambda_T_S")),
        })

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows([{c: ("" if r.get(c) is None else r.get(c)) for c in COLUMNS}
                          for r in rows])

    print(f"\n{len(rows)} quenchers -> {OUT.name}")

    if errors:
        absolute = [abs(e) for e in errors]
        rmse = math.sqrt(statistics.fmean(e * e for e in errors))
        print(f"\n  E_T, model against DFT, over {len(errors)} quenchers (kcal/mol):")
        print(f"    mean signed error {statistics.fmean(errors):+.2f}   "
              f"MAE {statistics.fmean(absolute):.2f}   RMSE {rmse:.2f}   "
              f"max |error| {max(absolute):.2f}")

        worst = sorted(rows, key=lambda r: -abs(r["E_T_error / kcal mol-1"] or 0))[:5]
        print("    largest disagreements:")
        for r in worst:
            print(f"      Q{r['quencher_id']:<3} {str(r['name'])[:28]:28} "
                  f"ml {r['E_T_ml / kcal mol-1']:6.1f}  dft {r['E_T_dft / kcal mol-1']:6.1f}  "
                  f"error {r['E_T_error / kcal mol-1']:+6.1f}  "
                  f"(model uncertainty {r['E_T_ml_uncertainty / kcal mol-1']:.1f})")

        # An uncertainty estimate is only useful if it is larger where the error is.
        within = sum(1 for r in rows
                     if r["E_T_error / kcal mol-1"] != ""
                     and abs(r["E_T_error / kcal mol-1"])
                     <= 2 * r["E_T_ml_uncertainty / kcal mol-1"])
        print(f"    within 2 sigma of the model's own uncertainty: {within}/{len(errors)}")


if __name__ == "__main__":
    main()
