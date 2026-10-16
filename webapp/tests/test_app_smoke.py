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


def test_only_the_screened_photocatalysts_carry_potentials(client):
    """The excited-state potentials were computed for the screened ones only."""
    for pc in client.get("/api/photocatalysts").json()["photocatalysts"]:
        values = (pc["E_PCplus_PCstar"], pc["E_PCstar_PCminus"])
        if pc["id"].startswith("OPC"):
            assert values == (None, None), pc["id"]
            assert pc["smiles"], f"{pc['id']} has no SMILES to draw it from"
        else:
            assert all(isinstance(v, float) for v in values), pc["id"]
            assert pc["E_PCplus_PCstar"] < pc["E_PCstar_PCminus"], pc["id"]
            assert pc["smiles"] == "", f"{pc['id']}: the metal complexes carry no SMILES"


def test_every_photocatalyst_is_drawn_or_named(client):
    """The drawing <id>.png wherever the file exists, else the SMILES; neither gives a 404
    the page turns into the compound name, never a 500."""
    import importlib.resources as ir

    drawings = ir.files("webapp.data").joinpath("pc_structures")
    for pc in client.get("/api/photocatalysts").json()["photocatalysts"]:
        response = client.post("/api/structure", json={"smiles": pc["smiles"], "name": pc["id"]})
        if drawings.joinpath(f"{pc['id']}.png").is_file():
            assert response.headers["content-type"] == "image/png", pc["id"]
        elif pc["smiles"]:
            assert response.headers["content-type"].startswith("image/svg"), pc["id"]
        else:
            assert response.status_code == 404, pc["id"]


def test_the_structure_name_is_an_id_not_a_path(client):
    response = client.post("/api/structure", json={"smiles": "", "name": "../photocatalysts"})
    assert response.status_code == 404


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


def test_marcus_gives_the_driving_force_and_the_barriers_only(client):
    """No lifetime goes in and no lifetime threshold or verdict comes out."""
    result = client.post("/api/marcus/compute", json=PAIR).json()
    assert set(result) == {"dG", "dG_activation", "dG_activation_backEnT"}
    assert result["dG"] < 0 and result["dG_activation"] > 0
    assert client.post("/api/marcus/compute",
                       json={**PAIR, "pc_tau0_ns": 2000.0}).status_code == 422


def test_the_model_was_never_called(client, stub):
    assert stub.calls == 0, "a route ran a forward pass -- the stub was bypassed"
