# webapp — EnTdecker

EnTdecker puts the model in a browser: draw or paste a quencher and get its predicted
triplet energy and reorganization energies (λS0-T1, the S0 surface at the T1 geometry,
and λT1-S0, the T1 surface at the S0 geometry) with their uncertainties, the predicted
distributions behind them and its spin-population map; pick a photocatalyst of the
library and get the driving force ΔG_EnT, the asymmetric Marcus barriers and the parabolas
of the pair. It takes no lifetime and gives no verdict: the barrier is the answer.

## The photocatalyst library

`data/photocatalysts.csv` is the photocatalyst library of the paper: the photocatalysts
of the Stern–Volmer screen (PC<n>, the paper's new ids) and the organic photocatalysts that
were computed but not screened (OPC<n>). It is generated, never edited by hand:

```
uv run webapp/scripts/build_photocatalysts.py
```

reads the repository's `data/` and takes each quantity from one source, the one the
benchmark uses: E_T, λS0-T1 and λT1-S0 from DFT, and the excited-state potentials
E(PC⁺/PC*) and E(PC*/PC⁻) from DFT (V vs SCE), which are shown but enter no calculation.
The organic photocatalysts have no potentials. Rerun the generator after any change to
those tables; `tests/test_photocatalyst_library.py` fails while the packaged table and
`data/` disagree, and checks that the app gives every benchmark pair the benchmark's driving
force and barrier.

A photocatalyst is drawn from `data/pc_structures/<id>.png` wherever that file exists;
adding or replacing a drawing needs no rerun. Without one, an organic photocatalyst is
drawn from its SMILES (read off its DFT geometry), and a metal complex, which has no SMILES,
is shown by its compound name. `scripts/` is not part
of the package: it has no `__init__.py`, is not installed, and nothing in the application
imports it.

The API speaks the paper's λ names (`lambda_S0_T1`, `lambda_T1_S0`) for everything
`webapp` defines. What `ml` exposes keeps its own names: the prediction keys `lamda_S_T`
(S0 surface) and `lamda_T_S` (T1 surface) and the arguments of `ml.energy.marcus_ent`,
which pair them into λforward = λT1-S0(PC) + λS0-T1(Q) and λreverse = λS0-T1(PC) + λT1-S0(Q).

This directory *is* the `webapp` package. It is a consumer of `ml` and nothing more: the
model, the checkpoint and the Marcus arithmetic all live on the other side of
`ml.prediction` and `ml.energy`, and nothing here recomputes any of them. The arrow
points one way — `webapp` imports `ml`, `ml` never imports `webapp`.

```
app.py              the FastAPI application, served as `webapp.app:app`
photocatalysts.py   the packaged photocatalyst library
smiles_utils.py     SMILES validation and canonicalization
plots/              Marcus, mode-distribution and spin-population figures
templates/          index.html, the single page
static/ketcher/     the EPAM Ketcher structure editor, as built for the browser
data/               photocatalysts.csv and the structure drawings, by photocatalyst id
scripts/            build_photocatalysts.py, the generator of photocatalysts.csv (not packaged)
tests/              a smoke test, the library against data/ and the benchmark, and the rule
                    that the dependency never reverses
```

`webapp` and `ml` are both built from the repository's single root `pyproject.toml`. The
`webapp` extra takes the `ml` extra whole and adds only what serving the page needs.

## Running it locally

```
uv sync --extra webapp
uv run uvicorn webapp.app:app --reload --port 5000
```

For predictions from Python or a terminal instead of a browser, use `ml.predict` or the
`ent-ml` command — same model, no web server.

The deployment (container image and cluster manifests) is site-specific and not part of
this repository.
