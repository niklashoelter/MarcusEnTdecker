"""Is the packaged photocatalyst table the paper's, and does the app score pairs as the
benchmark does?

Two checks against the rest of the repository, so both skip where it is absent (an
installed package, a container image):

- data/photocatalysts.csv is exactly what scripts/build_photocatalysts.py builds from
  data/, so a hand edit or a stale table after a data change fails here.
- For every scored pair of the benchmark and both quencher sources (DFT, ML), the app's
  /api/marcus/compute with the packaged photocatalyst gives the benchmark's driving force,
  asymmetric barrier, t_PC and two-gate verdict at the fixed contact distance of 7 A
  (`twogate_lifetime_r7A` in benchmark/further_analysis/metrics_extended.py). The pair
  table holds energies at 2 decimals, hence the tolerance of half a unit in the last place.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="needs the webapp extra: uv sync --extra webapp")
pytest.importorskip("httpx", reason="fastapi.testclient is built on httpx")

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "webapp" / "scripts" / "build_photocatalysts.py"
PAIRS = ROOT / "benchmark" / "further_analysis" / "results" / "metrics_extended" / "pairs.csv"
ROUNDED = 0.005 + 1e-9          # half the last place of a value written at 2 decimals
KC = "kcal mol-1"


def _generator():
    spec = importlib.util.spec_from_file_location("build_photocatalysts", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(a, b, abs_tol=1e-9)
    return a == b


def test_the_packaged_table_is_what_the_generator_builds(client):
    pytest.importorskip("openpyxl")
    generator = _generator()
    if not (generator.CALC.exists() and generator.EXP.exists() and generator.OPC_TABLE.exists()):
        pytest.skip("the repository's data/ is not here")
    built = generator.screened() + generator.organic()
    served = client.get("/api/photocatalysts").json()["photocatalysts"]
    assert [pc["id"] for pc in served] == [row["id"] for row in built]
    for row, pc in zip(built, served):
        differ = {k: (v, pc[k]) for k, v in row.items() if not _same(v, pc[k])}
        assert not differ, f"{row['id']}: table vs data/ {differ}; rerun {GENERATOR.name}"


def test_the_app_scores_every_benchmark_pair_as_the_benchmark_does(client):
    pd = pytest.importorskip("pandas")
    if not PAIRS.exists():
        pytest.skip("the benchmark results are not here")
    pairs = pd.read_csv(PAIRS)
    library = {pc["id"]: pc for pc in client.get("/api/photocatalysts").json()["photocatalysts"]}
    assert set(pairs["pc"]) <= set(library)

    mismatches = []
    for source in ("dft", "ml"):
        for _, pair in pairs.iterrows():
            pc = library[pair["pc"]]
            assert pc["tau0_ns"] == pytest.approx(pair["tau0 / ns"]), pair["pc"]
            body = {"pc_adiabatic_E_T": pc["E_T"],
                    "pc_lambda_S0_T1": pc["lambda_S0_T1"],
                    "pc_lambda_T1_S0": pc["lambda_T1_S0"],
                    "quencher_E_T": pair[f"E_T_q_{source} / {KC}"],
                    "quencher_lambda_S0_T1": pair[f"lambda_S0_T1_q_{source} / {KC}"],
                    "quencher_lambda_T1_S0": pair[f"lambda_T1_S0_q_{source} / {KC}"],
                    "pc_tau0_ns": pc["tau0_ns"]}
            with contextlib.redirect_stdout(io.StringIO()):   # the symmetric-fallback note
                app = client.post("/api/marcus/compute", json=body).json()
            expected = {"dG": pair[f"dG_EnT_{source} / {KC}"],
                        "dG_activation": pair[f"barrier_asym_{source} / {KC}"],
                        "t_PC": pair[f"t_PC_7A / {KC}"]}
            off = {k: (app[k], v) for k, v in expected.items() if abs(app[k] - v) > ROUNDED}
            if app["two_gate"] != bool(pair[f"pred_twogate_lifetime_r7A_{source}"]):
                off["two_gate"] = (app["two_gate"], not app["two_gate"])
            if off:
                mismatches.append((source, pair["pc"], pair["quencher"], off))
    assert not mismatches, f"{len(mismatches)} pairs differ, e.g. {mismatches[:5]}"
