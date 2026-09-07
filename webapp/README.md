# webapp — EnTdecker

EnTdecker is the public face of the models: draw a photocatalyst and a quencher, get a
predicted rate constant with its uncertainty and a Marcus reaction-coordinate plot.

The application code is not separate from the models. It lives inside the vendored
package at `../ml/upstream/pyrates/webapp/` — a FastAPI app (`fastapi_app.py`), the
science layer (`science_utils/`), the Ketcher structure editor served from
`static/ketcher/`, and the trained checkpoint at `data/model.pt`. This directory holds
only what is needed to *ship* it.

```
Dockerfile   python:3.12-slim, installs ml/upstream/, uvicorn on :5000
yaml/        Kubernetes manifests for the `entdecker` namespace
```

The pod runs two containers: the FastAPI application, and an nginx sidecar that proxies
8080 to 5000 and serves the health endpoint the readiness and liveness probes use.

## Running it locally

```
uv sync --extra ml
uv run uvicorn pyrates.webapp.fastapi_app:app --reload --port 5000
```

For predictions from Python instead of a browser, use `../ml/predict.py` — same model,
no web server.

## Building and deploying

Build from the repository root, not from this directory, so the context includes
`ml/upstream/`:

```
docker build -f webapp/Dockerfile -t ghcr.io/le-schlo/pyrates/entdecker:<tag> .
docker push ghcr.io/le-schlo/pyrates/entdecker:<tag>

wwukube login
kubectl apply -n entdecker -f webapp/yaml/
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
to tell them apart. It now copies `ml/upstream/`, pinned by this repository's commit,
with the upstream provenance recorded in `../ml/upstream/PROVENANCE.md`.

**Still to fix: the deployed tag is `:latest`.** `deployment.yaml` pairs that floating
tag with `imagePullPolicy: Always`, so the running application is whatever was pushed
most recently. For a paper that cites a live web tool, pin the manifest to the same
image tag the paper names, so a reader visiting EnTdecker gets the model the paper
reports rather than whatever came after it.

`_cheatsheet_original.txt` is the original working notes this README was written from,
kept until the commands above have been confirmed against the live cluster.

## Licensing

The Ketcher assets under `static/ketcher/` are third-party (EPAM Ketcher) and carry
their own licence, which is not the MIT licence covering this repository. Check it
before any redistribution beyond running the service.
