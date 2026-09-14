"""
Quencher SMILES -> the graph MDN's triplet energy and reorganization energies.

Reads   ../compound_identification.xlsx                (ids, names, SMILES)
        ../../ml/data/model.pt                         (the trained checkpoint)
Writes  ../computational_data_processed/quencher_properties_ml.xlsx

    uv sync --extra ml
    uv run --extra ml python data/computational_data_scripts/quencher_ml.py

One row per quencher, with the same layout and headers as `quencher_properties_calc.xlsx`.
Each predicted value comes with the network's uncertainty. The model's own field names
for lambda_S0_T1 and lambda_T1_S0 are `lamda_S_T` and `lamda_T_S`.

The script needs the `ml` package (torch, torch-geometric, rdkit) and the checkpoint
ml/data/model.pt, so it is not a stage of `uv run reproduce.py data`. The workbook is in
git. Per-atom spin densities and mixture parameters are not written; `ml.prediction.predict`
returns both.
"""

import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA))

from compound_identification import QUENCHERS  # noqa: E402
from properties_workbook import Block, Column, field, read_table, write_workbook  # noqa: E402

ROOT = DATA.parent
CHECKPOINT = ROOT / "ml" / "data" / "model.pt"
OUT = DATA / "computational_data_processed" / "quencher_properties_ml.xlsx"
Q_DFT = DATA / "computational_data_raw" / "quencher_adiabatic_DFT" / "quencher_dft.csv"

IDENTIFICATION = "compound_identification.xlsx"
MODEL = CHECKPOINT.relative_to(ROOT).as_posix()

KCAL = "kcal mol-1"
SIGFIGS = 6

# (header stem, the model's field name, what it is)
PREDICTED = [
    ("E_T", "triplet_energy_adiabatic",
     "Adiabatic triplet energy, predicted from the structure alone."),
    ("lambda_S0_T1", "lamda_S_T",
     "S0 surface: E_S0 at the T1 geometry minus E_S0 at its own."),
    ("lambda_T1_S0", "lamda_T_S",
     "T1 surface: E_T1 at the S0 geometry minus E_T1 at its own."),
]

UNCERTAINTY = ("Spread of the mixture density network's predicted distribution for that "
               "value, the model's own uncertainty estimate.")


def sig(x, n=SIGFIGS):
    """Round to six significant figures."""
    if x is None or x == "":
        return None
    return 0.0 if float(x) == 0 else float(f"{float(x):.{n}g}")


def predict_all(smiles):
    """{smiles: prediction} for every structure, in one pass over the model."""
    try:
        from ml.prediction import predict
    except ImportError as exc:
        sys.exit(f"{exc}\n\nthis script needs the `ml` package and its environment:\n"
                 f"    uv sync --extra ml\n"
                 f"    uv run --extra ml python "
                 f"{Path(__file__).resolve().relative_to(ROOT).as_posix()}")
    if not CHECKPOINT.exists():
        sys.exit(f"missing checkpoint: {CHECKPOINT}")

    print(f"predicting {len(smiles)} structures with {MODEL} ...", flush=True)
    frame = predict(smiles)
    return {row["smiles"]: row for _, row in frame.iterrows()}


def quenchers():
    ordered = sorted(QUENCHERS)
    without = [q for q in ordered if not QUENCHERS[q]["smiles"]]
    if without:
        sys.exit(f"quenchers without a SMILES, cannot predict: {without}")

    predictions = predict_all(list(dict.fromkeys(QUENCHERS[q]["smiles"] for q in ordered)))
    absent = [q for q in ordered if QUENCHERS[q]["smiles"] not in predictions]
    if absent:
        sys.exit(f"no prediction returned for: {absent}")

    # quencher id -> {column name: value}, the table the columns below read.
    table = {}
    for q in ordered:
        row = predictions[QUENCHERS[q]["smiles"]]
        table[q] = {}
        for name, source, _ in PREDICTED:
            table[q][name] = sig(row[source])
            table[q][f"{name}_uncertainty"] = sig(row[f"{source}_uncertainty"])

    def col(header, name, source, description):
        return Column(header, description, f"{MODEL}: {source}", field(table, name))

    predicted_columns = []
    for name, source, description in PREDICTED:
        predicted_columns.append(
            col(f"{name} / {KCAL}", name, source, description))
        predicted_columns.append(
            col(f"{name}_uncertainty / {KCAL}", f"{name}_uncertainty",
                f"{source}_uncertainty", UNCERTAINTY))

    blocks = [
        Block("Identity", [
            Column("quencher_id", "Quencher number.",
                   f"{IDENTIFICATION}: Quenchers/quencher_id", lambda i: i),
            Column("quencher_name", "Label used throughout, Q<n>.",
                   f"{IDENTIFICATION}: Quenchers/quencher_id", lambda i: f"Q{i}"),
            Column("compound", "Name derived from the SMILES, as in the other workbooks.",
                   f"{IDENTIFICATION}: Quenchers/name", lambda i: QUENCHERS[i]["name"]),
            Column("smiles", "Structure; the only input to the prediction.",
                   f"{IDENTIFICATION}: Quenchers/smiles", lambda i: QUENCHERS[i]["smiles"]),
        ]),
        Block(f"Graph MDN prediction / {KCAL}", predicted_columns),
    ]
    note = """
        One row per quencher, predicted from the SMILES alone by the graph mixture density
        network in ml/data/model.pt. The same three quantities from DFT are in
        quencher_properties_calc.xlsx under the same headers.
        Written by computational_data_scripts/quencher_ml.py, which needs the ml extra and
        is not a stage of reproduce.py.
        Read with pandas as: pd.read_excel(path, header=1), then drop the 'Unnamed' columns.
    """
    return ordered, blocks, note, table


def report(table):
    """Print how the predicted E_T compares with the DFT values."""
    if not Q_DFT.exists():
        return
    dft = read_table(Q_DFT, "quencher_id")
    errors = [(q, row["E_T"] - float((dft.get(q) or {}).get("E_T")))
              for q, row in table.items()
              if row["E_T"] is not None and (dft.get(q) or {}).get("E_T") not in ("", None)]
    if not errors:
        return

    by_size = sorted((abs(e), q) for q, e in errors)
    mean = sum(e for _, e in errors) / len(errors)
    mae = sum(a for a, _ in by_size) / len(by_size)
    rmse = (sum(e * e for _, e in errors) / len(errors)) ** 0.5
    print(f"  E_T against the DFT values, {len(errors)} quenchers (kcal/mol): "
          f"mean {mean:+.2f}, MAE {mae:.2f}, RMSE {rmse:.2f}")
    print("    furthest apart: "
          + ", ".join(f"Q{q} {a:.1f}" for a, q in reversed(by_size[-3:])))


def main():
    ids, blocks, note, table = quenchers()
    write_workbook(OUT, "Quenchers", ids, blocks, note)
    print(f"{len(ids)} quenchers x {sum(len(b.columns) for b in blocks)} columns "
          f"-> {OUT.name}")
    report(table)


if __name__ == "__main__":
    main()
