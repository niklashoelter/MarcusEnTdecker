# webapp — EnTdecker

EnTdecker is the public face of the models: draw a photocatalyst and a quencher, get a
predicted rate constant with its uncertainty and a Marcus reaction-coordinate plot.

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
static/ketcher/     the EPAM Ketcher structure editor
data/               photocatalysts.csv and the structure images
tests/              a smoke test, and the rule that the dependency never reverses
deploy/             Dockerfile and Kubernetes manifests — untracked, see below
```

`webapp` and `ml` are both built from the repository's single root `pyproject.toml`. The
`webapp` extra takes the `ml` extra whole and adds only what serving the page needs.

The pod runs two containers: the FastAPI application, and an nginx sidecar that proxies
8080 to 5000 and serves the health endpoint the readiness and liveness probes use.

## Running it locally

```
uv sync --extra webapp
uv run uvicorn webapp.app:app --reload --port 5000
```

For predictions from Python or a terminal instead of a browser, use `../ml/predict.py`
or the `ent-ml` command — same model, no web server.

## Building and deploying

`deploy/` is not in git. It holds the image definition and the cluster manifests, which
are site-specific and not part of the published record, so `.gitignore` keeps the whole
directory out; the commands below assume a checkout that still has the files on disk.

Build from the repository root, not from this directory, so the context includes
`pyproject.toml`, `ml/` and `webapp/`:

```
docker build -f webapp/deploy/Dockerfile -t ghcr.io/le-schlo/pyrates/entdecker:<tag> .
docker push ghcr.io/le-schlo/pyrates/entdecker:<tag>

wwukube login
kubectl apply -n entdecker -f webapp/deploy/yaml/
kubectl get -n entdecker pods
```

`wwukube` is the university cluster helper and is not committed here — it is a 3.5 MB
site-specific binary with its own distribution channel.

Rolling the deployment by hand:

```
kubectl delete -n entdecker --all deployments
kubectl delete -n entdecker --all service
kubectl delete -n entdecker --all pods
```

## What changed, and what still needs to

**Fixed: the image no longer clones a branch at build time.** It used to run
`git clone --branch fastapi-webapp`, which meant the image contents depended on *when*
it was built — two builds of the same tag could serve two different models with no way
to tell them apart. It now copies `pyproject.toml`, `ml/` and `webapp/`, pinned by this
repository's commit, with the provenance of the model code recorded in `../ml/README.md`.

**Still to fix: the deployed tag is `:latest`.** `deployment.yaml` pairs that floating
tag with `imagePullPolicy: Always`, so the running application is whatever was pushed
most recently. For a paper that cites a live web tool, pin the manifest to the same
image tag the paper names, so a reader visiting EnTdecker gets the model the paper
reports rather than whatever came after it.

`deploy/_cheatsheet_original.txt` is the original working notes this README was written
from, kept until the commands above have been confirmed against the live cluster.

## Licensing

The Ketcher assets under `static/ketcher/` are third-party (EPAM Ketcher) and carry
their own licence, which is not the MIT licence covering this repository. Check it
before any redistribution beyond running the service.
