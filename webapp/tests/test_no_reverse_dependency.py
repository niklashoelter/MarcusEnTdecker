"""The dependency runs one way: data -> ml -> webapp, and never back.

`ml` is the models and the prediction API; `webapp` is EnTdecker built on top of it.
So `webapp` may import `ml`, and `ml` may never import `webapp`.

Neither package may read anything under the repository's `data/` or `benchmark/`
directories either. Those are scripts and measurements, not a library: they are
regenerated, rearranged and partly kept out of git, and a package that reached into
them would stop working the moment it was installed anywhere but this checkout. The
packages carry their own resources instead -- `ml/data/model_final.pt`, `webapp/data/` --
and reach them through `importlib.resources`, not through a path.

Inside `ml`, `ml.training` is offline tooling -- dataset building, splits, the training
loop, metrics -- and is left out of the wheel. Nothing else in `ml` or `webapp` may
import it, or the installed package would break.

This is the rule that must never regress, which is why it is a test and not a note in
a README.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Directories excluded from the walk: the test suites themselves, which name the
# forbidden packages and paths for a living.
EXCLUDED = {("ml", "tests"), ("webapp", "tests")}

FORBIDDEN_PACKAGES = ("data", "benchmark")

# The scripts' own module names, so that a sys.path insert followed by, say,
# `import pairs` or `import properties_workbook` is caught as well.
SCRIPT_MODULES = {p.stem for d in FORBIDDEN_PACKAGES for p in (ROOT / d).rglob("*.py")
                  if "computational_data_raw" not in p.parts} | {"reproduce"}

# A path literal is forbidden if one of its segments is exactly `data` or `benchmark`,
# or if it is exactly that word. `ml.data` and `webapp.data` are module names, not
# paths, and never match; neither does `new_data/` or a `data:image/png;base64,` URI.
FORBIDDEN_SEGMENTS = {"data", "benchmark"}

# Unmistakable artefacts of the repository's data/ directory. Checked only against
# literals that look like a filename or a path, so that `photocatalyst_properties` --
# a function in ml/predict.py -- is not mistaken for the workbook of the same name.
FORBIDDEN_NAMES = (
    "exp_dataset",
    "compound_identification",
    "computational_data_processed",
    "computational_data_raw",
    "experimental_data_processed",
    "experimental_data_raw",
    "quencher_properties",
    "photocatalyst_properties",
    "stern_volmer",
    "properties_workbook",
)


def _modules(*packages):
    """Every *.py under the given package directories, test suites excluded."""
    found = []
    for package in packages:
        for path in sorted((ROOT / package).rglob("*.py")):
            parts = path.relative_to(ROOT).parts
            if any(parts[: len(excluded)] == excluded for excluded in EXCLUDED):
                continue
            found.append(path)
    assert found, f"no modules found under {packages} -- the walk is broken, not clean"
    return found


def _ids(paths):
    return [p.relative_to(ROOT).as_posix() for p in paths]


ML_MODULES = _modules("ml")
ALL_MODULES = _modules("ml", "webapp")
RUNTIME_MODULES = [p for p in ALL_MODULES
                   if p.relative_to(ROOT).parts[:2] != ("ml", "training")]


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_roots(tree):
    """Top-level package name of every absolute import in the module."""
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _imported_modules(tree):
    """Dotted name of every absolute import; `from a import b` counts as `a` and `a.b`."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


DATA_SUFFIXES =(".csv", ".tsv", ".xlsx", ".xls", ".gz", ".parquet", ".json", ".txt")


def _docstrings(tree):
    """ids of the docstring nodes, which are prose about paths and not paths."""
    out = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            out.add(id(body[0].value))
    return out


def _string_literals(tree):
    skip = _docstrings(tree)
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in skip]


def _looks_like_a_path(literal):
    return "/" in literal or "\\" in literal or literal.lower().endswith(DATA_SUFFIXES)


@pytest.mark.parametrize("path", ML_MODULES, ids=_ids(ML_MODULES))
def test_ml_never_imports_webapp(path):
    """ml is the library; webapp is its consumer. The arrow points one way."""
    assert "webapp" not in _imported_roots(_tree(path)), (
        f"{path.relative_to(ROOT).as_posix()} imports webapp. ml must not know that "
        f"EnTdecker exists -- move whatever is shared down into ml."
    )


@pytest.mark.parametrize("path", RUNTIME_MODULES, ids=_ids(RUNTIME_MODULES))
def test_runtime_never_imports_training(path):
    """ml.training is not installed with the package; nothing that is may need it."""
    offending = sorted(name for name in _imported_modules(_tree(path))
                       if name == "ml.training" or name.startswith("ml.training."))
    assert not offending, (
        f"{path.relative_to(ROOT).as_posix()} imports {offending}. ml.training is "
        f"offline tooling left out of the wheel -- move whatever is shared into ml.utils "
        f"or ml.model."
    )


@pytest.mark.parametrize("path", ALL_MODULES, ids=_ids(ALL_MODULES))
def test_no_package_imports_the_scripts(path):
    """`data/` and `benchmark/` are scripts, not importable packages."""
    offending = sorted(_imported_roots(_tree(path)) & (set(FORBIDDEN_PACKAGES) | SCRIPT_MODULES))
    assert not offending, (
        f"{path.relative_to(ROOT).as_posix()} imports {offending}. Those are the "
        f"repository's script directories, not libraries the packages may depend on."
    )


@pytest.mark.parametrize("path", ALL_MODULES, ids=_ids(ALL_MODULES))
def test_no_package_reads_data_or_benchmark(path):
    """No module under ml/ or webapp/ names a path in data/ or benchmark/."""
    offending = []
    for literal in _string_literals(_tree(path)):
        normalized = literal.replace("\\", "/")
        segments = normalized.split("/")
        if set(segments) & FORBIDDEN_SEGMENTS:
            offending.append(literal)
        elif (_looks_like_a_path(normalized)
              and any(name in normalized for name in FORBIDDEN_NAMES)):
            offending.append(literal)
    assert not offending, (
        f"{path.relative_to(ROOT).as_posix()} names {offending}, which reaches into "
        f"data/ or benchmark/. Package resources belong inside the package and are "
        f"read with importlib.resources -- see ml/prediction.py."
    )
