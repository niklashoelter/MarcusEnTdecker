"""Dependency rules: ml -> webapp, never back.

- `ml` never imports `webapp`.
- Neither package imports or reads anything under `data/` or `benchmark/`; resources
  (`ml/data/model_final.pt`, `webapp/data/`) via `importlib.resources`.
- `ml.training` (not in the wheel) is imported by nothing else in `ml` or `webapp`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Excluded from the walk: the test suites and webapp/scripts (offline generator, reads
# data/; checked by test_the_generator_stays_outside_the_package).
EXCLUDED = {("ml", "tests"), ("webapp", "tests"), ("webapp", "scripts")}

FORBIDDEN_PACKAGES = ("data", "benchmark")

# Script module names (catches `import pairs` after a sys.path insert).
SCRIPT_MODULES = {p.stem for d in FORBIDDEN_PACKAGES for p in (ROOT / d).rglob("*.py")
                  if "computational_data_raw" not in p.parts} | {"reproduce"}

# Forbidden: a literal with a path segment exactly `data` or `benchmark`. Does not match
# `ml.data`, `new_data/` or a `data:image/png;base64,` URI.
FORBIDDEN_SEGMENTS = {"data", "benchmark"}

# File names from data/; checked only against path-like literals.
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
    """ids of the docstring nodes (not checked for paths)."""
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
    """ml never imports webapp."""
    assert "webapp" not in _imported_roots(_tree(path)), (
        f"{path.relative_to(ROOT).as_posix()} imports webapp. ml must not know that "
        f"EnTdecker exists -- move whatever is shared down into ml."
    )


@pytest.mark.parametrize("path", RUNTIME_MODULES, ids=_ids(RUNTIME_MODULES))
def test_runtime_never_imports_training(path):
    """No runtime module imports ml.training."""
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


def test_the_generator_stays_outside_the_package():
    """webapp/scripts is no package and is imported by no runtime module."""
    assert not (ROOT / "webapp" / "scripts" / "__init__.py").exists(), (
        "webapp/scripts has become a package; it would be installed and break the rule above")
    offending = {path.relative_to(ROOT).as_posix(): sorted(
                     name for name in _imported_modules(_tree(path))
                     if name.startswith("webapp.scripts") or name == "build_photocatalysts")
                 for path in RUNTIME_MODULES}
    offending = {k: v for k, v in offending.items() if v}
    assert not offending, (
        f"{offending} import the offline generator of data/photocatalysts.csv. The "
        f"application reads the table, never the generator."
    )
