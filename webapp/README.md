# webapp — EnTdecker

EnTdecker serves the model in a browser: for a drawn or pasted quencher, the predicted
triplet energy and reorganization energies (λS0-T1, the S0 surface at the T1 geometry,
and λT1-S0, the T1 surface at the S0 geometry) with uncertainties, the predicted
distributions and the spin-population map; for a photocatalyst of the library, the
driving force ΔG_EnT, the asymmetric Marcus barriers and the parabolas of the pair. No
lifetime input.

## The photocatalyst library

`data/photocatalysts.csv`: the screened photocatalysts (PC<n>, new ids) and the organic
photocatalysts (OPC<n>). Generated, not edited by hand:

```
uv run webapp/scripts/build_photocatalysts.py
```

Sources (from the repository's `data/`): E_T, λS0-T1 and λT1-S0 from DFT; the
excited-state potentials E(PC⁺/PC*) and E(PC*/PC⁻) from DFT (V vs SCE), displayed only.
The organic photocatalysts have no potentials. `tests/test_photocatalyst_library.py`
checks the packaged table against `data/` and the app's driving force and barrier against
every benchmark pair.

Structures: `data/pc_structures/<id>.png` where present (no rerun needed); otherwise an
organic photocatalyst is drawn from its SMILES (from its DFT geometry) and a metal
complex is shown by name. `scripts/` is not part of the package.

λ names: the API uses `lambda_S0_T1`, `lambda_T1_S0`. `ml` keeps its own names: the
prediction keys `lamda_S_T` (S0 surface) and `lamda_T_S` (T1 surface) and the arguments
of `ml.energy.marcus_ent`, which pairs them into λforward = λT1-S0(PC) + λS0-T1(Q) and
λreverse = λS0-T1(PC) + λT1-S0(Q).

## Layout

This directory is the `webapp` package. It imports `ml` (`ml.prediction`, `ml.energy`);
`ml` never imports `webapp`.

```
app.py              the FastAPI application, served as `webapp.app:app`
photocatalysts.py   the packaged photocatalyst library
smiles_utils.py     SMILES validation and canonicalization
plots/              Marcus, mode-distribution and spin-population figures
templates/          index.html, the single page
static/ketcher/     the EPAM Ketcher structure editor, as built for the browser
data/               photocatalysts.csv and the structure drawings, by photocatalyst id
scripts/            build_photocatalysts.py, the generator of photocatalysts.csv (not packaged)
tests/              a smoke test, the library against data/ and the benchmark, and the
                    dependency direction
```

Built from the root `pyproject.toml`; the `webapp` extra includes the `ml` extra.

## Running it locally

```
uv sync --extra webapp
uv run uvicorn webapp.app:app --reload --port 5000
```

Without a browser: `ml.predict` or the `ent-ml` command.

The deployment (container image and cluster manifests) is not part of this repository.
