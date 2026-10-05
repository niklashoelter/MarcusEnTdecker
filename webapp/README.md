# webapp — EnTdecker

EnTdecker puts the model in a browser: draw or paste a quencher and get its predicted
triplet energy and reorganization energies with their uncertainties, the predicted
distributions behind them and its spin-population map; pick a photocatalyst from a short
table and get the driving force, the Marcus barriers and a reaction-coordinate plot for
the pair.

This directory *is* the `webapp` package. It is a consumer of `ml` and nothing more: the
model, the checkpoint and the Marcus arithmetic all live on the other side of
`ml.prediction` and `ml.energy`, and nothing here recomputes any of them. The arrow
points one way — `webapp` imports `ml`, `ml` never imports `webapp`.

```
app.py              the FastAPI application, served as `webapp.app:app`
photocatalysts.py   the packaged photocatalyst table
smiles_utils.py     SMILES validation and canonicalization
plots/              Marcus, mode-distribution and spin-population figures
templates/          index.html, the single page
static/ketcher/     the EPAM Ketcher structure editor, as built for the browser
data/               photocatalysts.csv and the structure images
tests/              a smoke test, and the rule that the dependency never reverses
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
