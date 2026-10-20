"""The model stub must not leak into later tests or application imports."""

from contextlib import contextmanager
import importlib
import sys

import pytest

from webapp.tests.conftest import client as client_fixture


@pytest.mark.parametrize("preloaded", [False, True])
def test_client_restores_modules_and_package_attributes(monkeypatch, preloaded):
    import ml
    import webapp

    importlib.import_module("ml.prediction")
    importlib.import_module("webapp.app")
    if not preloaded:
        for name in ("ml.prediction", "webapp.app"):
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.delattr(ml, "prediction")
        monkeypatch.delattr(webapp, "app")

    missing = object()
    modules = {name: sys.modules.get(name, missing)
               for name in ("ml.prediction", "webapp.app")}
    attributes = {(package, name): vars(package).get(name, missing)
                  for package, name in ((ml, "prediction"), (webapp, "app"))}
    with contextmanager(client_fixture.__wrapped__)(lambda smiles: None) as client:
        assert client.get("/").status_code == 200

    for name, previous in modules.items():
        assert sys.modules.get(name, missing) is previous
    for (package, name), previous in attributes.items():
        assert vars(package).get(name, missing) is previous
