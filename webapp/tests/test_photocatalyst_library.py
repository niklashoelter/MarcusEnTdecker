"""Packaged photocatalyst table and app against the repository (skip without it).

- data/photocatalysts.csv equals what scripts/build_photocatalysts.py builds.
- /api/marcus/compute gives the driving force and asymmetric barrier of every pair in
  benchmark/further_analysis/results/metrics_extended/pairs.csv (DFT and ML quencher),
  within half a unit of the 2nd decimal.
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
    if not (generator.CALC.exists() and generator.OPC_TABLE.exists()):
        pytest.skip("the repository's data/ is not here")
    built = generator.screened() + generator.organic()
    served = client.get("/api/photocatalysts").json()["photocatalysts"]
    assert [pc["id"] for pc in served] == [row["id"] for row in built]
    for row, pc in zip(built, served):
        differ = {k: (v, pc[k]) for k, v in row.items() if not _same(v, pc[k])}
        assert not differ, f"{row['id']}: table vs data/ {differ}; rerun {GENERATOR.name}"


def test_the_app_gives_every_benchmark_pair_its_driving_force_and_barrier(client):
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
            body = {"pc_adiabatic_E_T": pc["E_T"],
                    "pc_lambda_S0_T1": pc["lambda_S0_T1"],
                    "pc_lambda_T1_S0": pc["lambda_T1_S0"],
                    "quencher_E_T": pair[f"E_T_q_{source} / {KC}"],
                    "quencher_lambda_S0_T1": pair[f"lambda_S0_T1_q_{source} / {KC}"],
                    "quencher_lambda_T1_S0": pair[f"lambda_T1_S0_q_{source} / {KC}"]}
            with contextlib.redirect_stdout(io.StringIO()):   # the symmetric-fallback note
                app = client.post("/api/marcus/compute", json=body).json()
            expected = {"dG": pair[f"dG_EnT_{source} / {KC}"],
                        "dG_activation": pair[f"barrier_asym_{source} / {KC}"]}
            off = {k: (app[k], v) for k, v in expected.items() if abs(app[k] - v) > ROUNDED}
            if off:
                mismatches.append((source, pair["pc"], pair["quencher"], off))
    assert not mismatches, f"{len(mismatches)} pairs differ, e.g. {mismatches[:5]}"
