"""ent-ml: the command line over `ml.predict`.

    ent-ml predict SMILES ...   triplet energy and both reorganization energies
    ent-ml rate --quencher ...  driving force, Marcus barrier and k_EnT for one pair
    ent-ml info                 which checkpoint is loaded, and what it can encode

Reads SMILES from the command line or from a file, and the photocatalyst table from a
path the caller gives -- this package ships neither. Writes a table, CSV or JSON, to
stdout or to `--out`. Torch is imported only where it is needed, so `--help` is free.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
from pathlib import Path

from ml.predict import RatePredictor, photocatalyst_properties


def _valid(smiles: list[str]) -> list[str]:
    """The structures, refused before the model is loaded if any is not a SMILES."""
    from rdkit import Chem

    unparsed = [s for s in smiles if Chem.MolFromSmiles(s) is None]
    if unparsed:
        raise ValueError(f"not a valid SMILES: {', '.join(unparsed)}")
    return smiles


def _quiet_model():
    """Loading the checkpoint prints progress; keep it off the output stream."""
    return contextlib.redirect_stdout(sys.stderr)


def _cell(value) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _emit(rows: list[dict], fmt: str, out: str | None) -> None:
    if fmt == "json":
        text = json.dumps(rows, indent=2)
    elif fmt == "csv":
        buffer = io.StringIO()
        # csv defaults to CRLF; the one newline convention keeps --out and stdout equal.
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        text = buffer.getvalue().strip("\n")
    else:
        columns = list(rows[0])
        cells = [[_cell(row[c]) for c in columns] for row in rows]
        widths = [max(len(c), *(len(row[i]) for row in cells))
                  for i, c in enumerate(columns)]
        header = [c.ljust(w) for c, w in zip(columns, widths)]
        body = [[v.ljust(w) for v, w in zip(row, widths)] for row in cells]
        text = "\n".join("  ".join(line).rstrip() for line in [header, *body])

    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _predict(args) -> None:
    if args.file:
        lines = Path(args.file).read_text(encoding="utf-8").splitlines()
        smiles = [line.strip() for line in lines if line.strip()]
    else:
        smiles = args.smiles
    if not smiles:
        raise ValueError("no SMILES given; pass them as arguments or with --file")

    with _quiet_model():
        predicted = RatePredictor().properties(_valid(smiles))

    rows = [
        {"smiles": p.smiles,
         "E_T": p.E_T, "E_T_uncertainty": p.E_T_uncertainty,
         "lambda_S_T": p.lambda_S_T, "lambda_S_T_uncertainty": p.lambda_S_T_uncertainty,
         "lambda_T_S": p.lambda_T_S, "lambda_T_S_uncertainty": p.lambda_T_S_uncertainty}
        for p in predicted
    ]
    _emit(rows, args.format, args.out)


def _rate(args) -> None:
    quencher = _valid([args.quencher])[0]
    photocatalyst = photocatalyst_properties(args.photocatalysts, args.photocatalyst)
    with _quiet_model():
        result = RatePredictor(lambda_scale=args.lambda_scale).predict(photocatalyst,
                                                                      quencher)
    _emit([{"photocatalyst": result.photocatalyst,
            "quencher": result.quencher,
            "dG_kcal": result.dG_kcal,
            "dG_barrier_kcal": result.dG_barrier_kcal,
            "k_ent": result.k_ent,
            "quenches": result.quenches}], args.format, args.out)


def _info(args) -> None:
    import importlib.resources as ir

    import torch
    from rdkit.Chem import GetPeriodicTable

    from ml.model import checkpoint as _checkpoint  # noqa: F401  aliases the pre-rename module path
    from ml.utils.graph_construction import ATOM_FEATURES

    path = Path(str(ir.files("ml.data").joinpath("model_final.pt")))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    elements = GetPeriodicTable()

    print(f"checkpoint  {path}")
    for key, label in (("epoch", "epoch"), ("val_loss", "val loss")):
        if key in checkpoint:
            print(f"{label:<11} {checkpoint[key]}")
    print("elements    " + " ".join(elements.GetElementSymbol(z)
                                    for z in ATOM_FEATURES["atomic_num"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ent-ml", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--format", choices=("table", "csv", "json"), default="table")
    shared.add_argument("--out", metavar="PATH", help="write here instead of to stdout")

    predict = commands.add_parser("predict", parents=[shared],
                                  help="properties of each structure, from the model")
    predict.add_argument("smiles", nargs="*")
    predict.add_argument("--file", metavar="PATH",
                         help="read SMILES from this file, one per line")
    predict.set_defaults(run=_predict)

    rate = commands.add_parser("rate", parents=[shared],
                               help="rate constant for one photocatalyst-quencher pair")
    rate.add_argument("--quencher", required=True, metavar="SMILES")
    rate.add_argument("--photocatalysts", required=True, metavar="PATH",
                      help="an .xlsx or .csv with one row per photocatalyst")
    rate.add_argument("--photocatalyst", metavar="ID",
                      help="new_id of the row to use; omit only if the table has one row")
    rate.add_argument("--lambda-scale", type=float, default=1.0, metavar="X",
                      help="scales the predicted reorganization energies (default 1.0)")
    rate.set_defaults(run=_rate)

    info = commands.add_parser("info", help="the loaded checkpoint and its vocabulary")
    info.set_defaults(run=_info)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.run(args)
    except KeyError as exc:  # its str() would quote the message
        sys.exit(f"ent-ml: {exc.args[0]}")
    except (OSError, ValueError) as exc:
        sys.exit(f"ent-ml: {exc}")


if __name__ == "__main__":
    main()
