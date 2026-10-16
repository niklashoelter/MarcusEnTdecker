"""Does EnTdecker still come up, and do its routes still answer?

A smoke test, not a test of the science: the model is stubbed out, so nothing here
runs a forward pass. What it checks is that `webapp.app` imports under its new module
path, that the page renders, and that an API route that needs no prediction answers.

It needs the `webapp` extra for fastapi, and the dev group for httpx, which fastapi's
test client is built on. Under `uv sync --extra ml` alone fastapi is missing and the whole
file skips; the dependency test beside it does not, because it only reads source.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="needs the webapp extra: uv sync --extra webapp")
pytest.importorskip("httpx", reason="fastapi.testclient is built on httpx")

# The `client` and `stub` fixtures (the app with the model stubbed out) are in conftest.py.


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
    assert {"id", "name", "E_T", "lambda_S0_T1", "lambda_T1_S0"} <= set(first)
    assert isinstance(first["E_T"], float)
    assert not any("lamda" in key for key in first), "the old lambda names are back"


def test_the_library_is_the_screened_photocatalysts_and_the_organic_ones(client):
    """PC1-PC27, the photocatalysts of the screen, then OPC1-OPC8, offered only here."""
    ids = [pc["id"] for pc in client.get("/api/photocatalysts").json()["photocatalysts"]]
    assert ids == [f"PC{n}" for n in range(1, 28)] + [f"OPC{n}" for n in range(1, 9)]


def test_only_the_screened_photocatalysts_carry_lifetime_and_potentials(client):
    """tau0 was measured and the potentials computed for the screened ones only."""
    for pc in client.get("/api/photocatalysts").json()["photocatalysts"]:
        values = (pc["tau0_ns"], pc["E_PCplus_PCstar"], pc["E_PCstar_PCminus"])
        if pc["id"].startswith("OPC"):
            assert values == (None, None, None), pc["id"]
            assert pc["smiles"], f"{pc['id']} has no SMILES to draw it from"
        else:
            assert all(isinstance(v, float) for v in values), pc["id"]
            assert pc["E_PCplus_PCstar"] < pc["E_PCstar_PCminus"], pc["id"]
            assert pc["smiles"] == "", f"{pc['id']}: the metal complexes carry no SMILES"


def test_every_photocatalyst_is_drawn_or_named(client):
    """A drawing or a SMILES gives a structure; neither gives a 404 the page turns into the
    compound name, never a 500."""
    for pc in client.get("/api/photocatalysts").json()["photocatalysts"]:
        response = client.post("/api/structure", json={"smiles": pc["smiles"], "name": pc["id"]})
        if pc["structure_image"]:
            assert response.headers["content-type"] == "image/png", pc["id"]
        elif pc["smiles"]:
            assert response.headers["content-type"].startswith("image/svg"), pc["id"]
        else:
            assert response.status_code == 404, pc["id"]


def test_the_structure_name_is_an_id_not_a_path(client):
    response = client.post("/api/structure", json={"smiles": "", "name": "../photocatalysts"})
    assert response.status_code == 404


def test_photocatalysts_carry_their_measured_lifetime(client):
    """tau0 where it was measured, None where it was not."""
    photocatalysts = client.get("/api/photocatalysts").json()["photocatalysts"]
    lifetimes = {pc["name"]: pc["tau0_ns"] for pc in photocatalysts}
    assert any(isinstance(t, float) and t > 0 for t in lifetimes.values())
    assert all(t is None or t > 0 for t in lifetimes.values())


PAIR = {"pc_adiabatic_E_T": 65.7, "pc_lambda_S0_T1": 10.3, "pc_lambda_T1_S0": 8.3,
        "quencher_E_T": 58.0, "quencher_lambda_S0_T1": 9.0, "quencher_lambda_T1_S0": 7.0}


def test_marcus_pairs_the_lambdas_as_ml_energy_does(client):
    """lambda_forward = lambda_T1_S0(PC) + lambda_S0_T1(Q): the route hands each surface to
    the argument of ml.energy that holds it."""
    from ml.energy.marcus_ent import _compute_marcus_core

    expected = _compute_marcus_core(65.7, 10.3, 8.3, 58.0, 9.0, 7.0)
    result = client.post("/api/marcus/compute", json=PAIR).json()
    assert result["dG_activation"] == pytest.approx(expected[1])
    swapped = {**PAIR, "pc_lambda_S0_T1": 8.3, "pc_lambda_T1_S0": 10.3}
    assert client.post("/api/marcus/compute", json=swapped).json()["dG_activation"] != (
        pytest.approx(expected[1]))


def test_marcus_refuses_the_old_lambda_names(client):
    old = {"pc_adiabatic_E_T": 65.7, "pc_lambda_S_T": 10.3, "pc_lambda_T_S": 8.3,
           "quencher_E_T": 58.0, "quencher_lambda_S_T": 9.0, "quencher_lambda_T_S": 7.0}
    assert client.post("/api/marcus/compute", json=old).status_code == 422


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
