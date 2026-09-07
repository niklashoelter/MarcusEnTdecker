# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ── Timezone ──────────────────────────────────────────────────────────────────
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        tzdata \
        libcairo2 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── Application ───────────────────────────────────────────────────────────────
#
# This copies ml/ out of the repository rather than cloning le-schlo/pyrates at
# build time, as it used to. Cloning a *branch* meant the image contents depended on
# when it was built, so two builds of the "same" image could serve two different
# models with no way to tell them apart. The copy here is pinned by whatever commit
# of this repository is checked out, and ml/README.md records which commit of
# le-schlo/pyrates the model code corresponds to.
#
# The whole of ml/ travels, not just the package: predict.py, the READMEs, tests/,
# examples/ and training_data/dataset.csv (9 MB). All inert at runtime. If image size
# starts to matter, add a .dockerignore for training_data/ and tests/ rather than
# splitting the directory.
#
# Build from the repository root so the context includes ml/:
#
#     docker build -f webapp/Dockerfile -t ghcr.io/le-schlo/pyrates/entdecker:<tag> .
#
# The trained checkpoint travels with the package at
# pyrates/webapp/data/model.pt (69 MB), so no weights are downloaded at runtime.

WORKDIR /app
COPY ml/ /app/

RUN pip install --no-cache-dir -e .

# ── FastAPI on port 5000 (nginx sidecar proxies 8080 → 5000) ─────────────────
EXPOSE 5000

CMD ["uvicorn", \
     "pyrates.webapp.fastapi_app:app", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--workers", "1", \
     "--timeout-keep-alive", "1200", \
     "--log-level", "info"]
