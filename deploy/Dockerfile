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
        git \
        libcairo2 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── Clone repo from GitHub ────────────────────────────────────────────────────

WORKDIR /app

RUN git clone --branch fastapi-webapp --single-branch --depth 1 \
    https://le-schlo@github.com/le-schlo/pyrates.git .

# ── Install package ───────────────────────────────────────────────────────────
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
