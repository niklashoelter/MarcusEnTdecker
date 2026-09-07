# deploy — the EnTdecker web application

EnTdecker is pyrates with a FastAPI front end. There is no separate application source
code: the image is built by cloning the `fastapi-webapp` branch of
[pyrates](https://github.com/le-schlo/pyrates) and serving
`pyrates.webapp.fastapi_app:app`.

This directory holds the deployment only — the image definition and the Kubernetes
manifests for the university cluster.

```
Dockerfile   python:3.12-slim, clones pyrates@fastapi-webapp, uvicorn on :5000
yaml/        Kubernetes manifests for the `entdecker` namespace
```

The pod runs two containers: the FastAPI application, and an nginx sidecar that proxies
8080 to 5000 and serves the health endpoint the readiness and liveness probes use.

## Building and deploying

```
docker build -t ghcr.io/le-schlo/pyrates/entdecker:latest .
docker push ghcr.io/le-schlo/pyrates/entdecker:latest

kubectl apply -n entdecker -f yaml/
kubectl get -n entdecker pods
```

Cluster access is through the `wwukube` helper, which is not committed here — it is a
3.5 MB university-specific binary with its own distribution channel. `wwukube login`
first.

Rolling the deployment by hand:

```
kubectl delete -n entdecker --all deployments
kubectl delete -n entdecker --all service
kubectl delete -n entdecker --all pods
```

## Two things to fix before publication

**The image tag is `:latest`.** The manifest sets `imagePullPolicy: Always` against a
floating tag, so the deployed application is whatever was last pushed. For a paper that
cites a live web application, the deployed image should be pinned to the same pyrates
release that `../ml/` pins, so the tool a reader visits matches the model the paper
reports.

**The Dockerfile clones the branch at build time**, which means the image contents
depend on when it was built. Same fix: clone a tag, not a branch.

`_cheatsheet_original.txt` is the original working notes this README was written from,
kept until the commands above have been confirmed against the live cluster.
