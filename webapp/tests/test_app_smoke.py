"""Does EnTdecker still come up, and do its routes still answer?

A smoke test, not a test of the science: the model is stubbed out, so nothing here
runs a forward pass. What it checks is that `webapp.app` imports under its new module
path, that the page renders, and that an API route that needs no prediction answers.

It needs the `webapp` extra, which brings fastapi and its test client's httpx. Under
`uv sync --extra ml` alone neither is installed and the whole file skips; the
dependency test beside it does not, because it only reads source.
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


def test_the_model_was_never_called(client, stub):
    assert stub.calls == 0, "a route ran a forward pass -- the stub was bypassed"
