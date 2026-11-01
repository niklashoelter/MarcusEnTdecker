"""Fixtures shared by the webapp tests: the app with the model stubbed out."""

from __future__ import annotations

import sys
import types

import pytest


class _Stub:
    """Stands in for `ml.prediction.predict` and counts the calls it never gets."""

    def __init__(self):
        self.calls = 0

    def __call__(self, smiles_list):
        self.calls += 1
        raise AssertionError("the webapp tests must not run a forward pass")


@pytest.fixture(scope="module")
def stub():
    return _Stub()


@pytest.fixture(scope="module")
def client(stub):
    """The app with `ml.prediction` stubbed in `sys.modules`; no `with` block, so the
    lifespan (model pre-warm) does not run."""
    pytest.importorskip("fastapi", reason="needs the webapp extra: uv sync --extra webapp")
    pytest.importorskip("httpx", reason="fastapi.testclient is built on httpx")
    from fastapi.testclient import TestClient

    fake = types.ModuleType("ml.prediction")
    fake.predict = stub

    import ml
    import webapp

    missing = object()
    attributes = {(package, name): vars(package).get(name, missing)
                  for package, name in ((ml, "prediction"), (webapp, "app"))}
    saved = {name: sys.modules[name] for name in ("ml.prediction", "webapp.app")
             if name in sys.modules}
    sys.modules["ml.prediction"] = fake
    sys.modules.pop("webapp.app", None)

    ml.prediction = fake
    try:
        from webapp.app import app

        yield TestClient(app)
    finally:
        sys.modules.pop("webapp.app", None)
        sys.modules.pop("ml.prediction", None)
        sys.modules.update(saved)
        for (package, name), previous in attributes.items():
            if previous is missing:
                vars(package).pop(name, None)
            else:
                setattr(package, name, previous)
