"""
Quencher SMILES -> the graph MDN's triplet energy and reorganization energies.

Reads   ../compound_identification.xlsx                (ids, names, SMILES)
        ../../ml/data/model.pt                         (the trained checkpoint)
Writes  ../computational_data_processed/quencher_properties_ml.xlsx

One row per quencher, in the same layout and headers as `quencher_properties_calc.xlsx`,
so the model's E_T, lambda_S_T and lambda_T_S sit beside the DFT ones column for column.
Each value carries the network's own uncertainty, which a DFT number has not.

IT NEEDS THE MODEL ENVIRONMENT, SO IT IS NOT A reproduce.py STAGE
    Everything else under data/ runs standalone under `uv run`. This needs the `ml`
    package, which pins torch, torch-geometric and rdkit -- gigabytes the rest of the
    pipeline has no use for -- so it is run with the ml extra:

        uv sync --extra ml
        uv run --extra ml python data/computational_data_scripts/quencher_ml.py

    `uv run reproduce.py data` rebuilds everything else here without installing torch, and
    the workbook is in git, so a clone has the predictions without loading the model. The
    checkpoint is the one thing this reaches outside data/ for: a model is not an
    instrument and its weights are not a measurement.

WHY NOT properties_calc.py
    That script only copies from tables the extraction scripts left in
    ../computational_data_raw/, and must stay runnable with `uv run` alone. A model has no
    such table -- no job directory, only a forward pass -- so prediction and workbook are
    one step here, and the `Column key` sheet names the checkpoint and field instead.

WHAT IS NOT CARRIED
    The per-atom spin densities and the mixture parameters behind each prediction: neither
    is a per-compound scalar. Re-run the model for them --
    `ml.prediction.predict` returns both.
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
    ("lambda_S_T", "lamda_S_T",
     "Singlet-surface reorganization energy, the model's counterpart to the DFT "
     "lambda_S_T."),
    ("lambda_T_S", "lamda_T_S",
     "Triplet-surface reorganization energy, the model's counterpart to the DFT "
     "lambda_T_S."),
]

UNCERTAINTY = ("Spread of the mixture density network's own distribution for that value. "
               "The model's estimate of its own confidence, not a DFT or measured error.")


def sig(x, n=SIGFIGS):
    """Six significant figures, the rounding used everywhere except quencher_dft.py."""
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
            Column("smiles", "The structure the prediction was made from, and its only "
                             "input.",
                   f"{IDENTIFICATION}: Quenchers/smiles", lambda i: QUENCHERS[i]["smiles"]),
        ]),
        Block(f"Graph MDN prediction / {KCAL}", predicted_columns),
    ]
    note = """
        One row per quencher, predicted from the SMILES alone by the graph mixture density
        network in ml/data/model.pt. Nothing measured and nothing calculated
        enters a value here: the same three quantities from DFT are in
        quencher_properties_calc.xlsx, column for column, and the two files are meant to be
        read side by side rather than merged. The per-atom spin densities and the mixture
        parameters the model also returns are not carried.
        Written by computational_data_scripts/quencher_ml.py, which needs the ml extra and
        is not a stage of reproduce.py.
        Read with pandas as: pd.read_excel(path, header=1), then drop the 'Unnamed' columns.
    """
    return ordered, blocks, note, table


def report(table):
    """The model against the DFT triplet energies. A check on the run, not an output."""
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
