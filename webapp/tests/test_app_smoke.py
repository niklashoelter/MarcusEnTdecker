"""Does EnTdecker still come up, and do its routes still answer?

A smoke test, not a test of the science: the model is stubbed out, so nothing here
runs a forward pass. What it checks is that `webapp.app` imports under its new module
path, that the page renders, and that an API route that needs no prediction answers.

It needs the `webapp` extra for fastapi, and the dev group for httpx, which fastapi's
test client is built on. Under `uv sync --extra ml` alone fastapi is missing and the whole
file skips; the dependency test beside it does not, because it only reads source.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("fastapi", reason="needs the webapp extra: uv sync --extra webapp")
pytest.importorskip("httpx", reason="fastapi.testclient is built on httpx")


class _Stub:
    """Stands in for `ml.prediction.predict` and counts the calls it never gets."""

    def __init__(self):
        self.calls = 0

    def __call__(self, smiles_list):
        self.calls += 1
        raise AssertionError("the smoke test must not run a forward pass")


@pytest.fixture(scope="module")
def stub():
    return _Stub()


@pytest.fixture(scope="module")
def client(stub):
    """The app with the model stubbed out, and no lifespan.

    `ml.prediction` is replaced in `sys.modules` before `webapp.app` is imported, so
    the checkpoint is never loaded and torch is never asked for a forward pass. The
    client is used outside a `with` block on purpose: entering one would run the
    lifespan, whose whole job is to pre-warm the model.
    """
    from fastapi.testclient import TestClient

    fake = types.ModuleType("ml.prediction")
    fake.predict = stub

    saved = {name: sys.modules[name] for name in ("ml.prediction", "webapp.app")
             if name in sys.modules}
    sys.modules["ml.prediction"] = fake
    sys.modules.pop("webapp.app", None)

    import ml

    ml.prediction = fake
    try:
        from webapp.app import app

        yield TestClient(app)
    finally:
        sys.modules.pop("webapp.app", None)
        sys.modules.pop("ml.prediction", None)
        sys.modules.update(saved)


def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "EnTdecker" in response.text


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_photocatalyst_api_answers(client):
    """The packaged photocatalyst table, served without touching the model."""
    response = client.get("/api/photocatalysts")
    assert response.status_code == 200

    photocatalysts = response.json()["photocatalysts"]
    assert photocatalysts, "the packaged photocatalysts.csv came back empty"
    first = photocatalysts[0]
    assert {"name", "E_T_adiabatic", "lamda_S_T", "lamda_T_S"} <= set(first)
    assert isinstance(first["E_T_adiabatic"], float)


def test_photocatalysts_carry_their_measured_lifetime(client):
    """tau0 where it was measured, None where it was not."""
    photocatalysts = client.get("/api/photocatalysts").json()["photocatalysts"]
    lifetimes = {pc["name"]: pc["tau0_ns"] for pc in photocatalysts}
    assert any(isinstance(t, float) and t > 0 for t in lifetimes.values())
    assert all(t is None or t > 0 for t in lifetimes.values())


PAIR = {"pc_adiabatic_E_T": 65.7, "pc_lambda_S_T": 10.3, "pc_lambda_T_S": 8.3,
        "quencher_E_T": 58.0, "quencher_lambda_S_T": 9.0, "quencher_lambda_T_S": 7.0}


def test_marcus_without_lifetime_gives_the_barrier_only(client):
    result = client.post("/api/marcus/compute", json=PAIR).json()
    assert result["dG"] < 0 and result["dG_activation"] > 0
    assert result["t_PC"] is None and result["two_gate"] is None


def test_marcus_with_lifetime_gives_the_two_gate_verdict(client):
    from ml.energy.two_gate import lifetime_barrier

    result = client.post("/api/marcus/compute", json={**PAIR, "pc_tau0_ns": 2000.0}).json()
    assert result["t_PC"] == pytest.approx(lifetime_barrier(2000.0))
    assert result["two_gate"] is (result["dG"] <= 0
                                  and result["dG_activation"] <= result["t_PC"])

    short = client.post("/api/marcus/compute", json={**PAIR, "pc_tau0_ns": 0.4}).json()
    assert short["t_PC"] is None and short["two_gate"] is False


def test_marcus_lifetime_barrier_follows_the_energy_unit(client):
    kcal = client.post("/api/marcus/compute", json={**PAIR, "pc_tau0_ns": 2000.0}).json()
    in_ev = {k: v / 23.060548868 for k, v in PAIR.items()}
    ev = client.post("/api/marcus/compute",
                     json={**in_ev, "unit": "eV", "pc_tau0_ns": 2000.0}).json()
    assert ev["t_PC"] == pytest.approx(kcal["t_PC"] / 23.060548868)
    assert ev["two_gate"] == kcal["two_gate"]


def test_marcus_refuses_a_lifetime_that_is_not_positive(client):
    response = client.post("/api/marcus/compute", json={**PAIR, "pc_tau0_ns": 0.0})
    assert response.status_code == 400


def test_the_model_was_never_called(client, stub):
    assert stub.calls == 0, "a route ran a forward pass -- the stub was bypassed"
