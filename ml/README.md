# ml — models and predictions

```
upstream/     pyrates, vendored verbatim from le-schlo/pyrates (MIT). See its PROVENANCE.md.
predict.py    the prediction API: two SMILES in, a rate constant out
```

The models are Leon Schlosser's and are not reimplemented here. `upstream/` is a
copy of his repository at a recorded commit, including the trained checkpoint the web
application serves (`upstream/pyrates/webapp/data/model.pt`, 69 MB), so that the paper
stays reproducible after the library moves on. **Do not edit anything in `upstream/`** —
fixes go upstream and come back as a re-vendor.

## Installing

```
uv sync --extra ml
```

This installs `upstream/` in editable mode from its own unmodified `pyproject.toml`,
which is what pins torch, torch-geometric and rdkit. It is deliberately not part of
the default install: the data pipeline needs none of it, and `uv run reproduce.py`
should not drag in a multi-gigabyte dependency tree.

## Predicting

```python
from ml.predict import RatePredictor

predictor = RatePredictor()
result = predictor.predict(photocatalyst=PC_SMILES, quencher=Q_SMILES)

result.dG_kcal          # driving force
result.dG_barrier_kcal  # Marcus activation barrier
result.k_ent            # rate constant, M-1 s-1
result.quenches         # above the 1e7 detection threshold
result.uncertainty      # from the mixture density network
```

`predict_many` takes an iterable of pairs and runs one forward pass per *structure*,
not per pair, so screening 48 quenchers against one photocatalyst costs 49 passes
rather than 96. There is no command-line interface; this is a library.

### What it composes

Nothing in `predict.py` reimplements the model or the theory. It is the seam:

```
upstream prediction.predict(smiles)   graph -> E_T, lambda_S_T, lambda_T_S
upstream marcus_theory                properties -> dG and the activation barrier
Balzani rate expression               barrier -> k, capped at the diffusion limit
```

Only the last step is written here, because upstream stops at the barrier. Keeping the
barrier upstream means the web application, the paper and this API compute it with the
same code.

The rate is `k_diff / (1 + exp(dG/RT) + 0.25 exp(dG‡/RT))`, which saturates at the
diffusion limit. A bare Eyring rate has no such ceiling and will return 1e14 M⁻¹ s⁻¹
for a strongly exergonic pair, so it is exposed as `k_eyring` for comparison but is
not what `k_ent` reports.

### λ-scaling

`RatePredictor(lambda_scale=...)` scales the predicted reorganization energies before
they reach the barrier. It defaults to 1.0 — raw prediction, nothing silently applied
— but 1.0 is known to over-estimate the barrier, and the benchmark finds an optimum
near 0.35–0.45. The reasoning is in `../benchmark/README.md`; it is a physical
correction, not a fitted parameter.

## Not yet run

`predict.py` has not been executed end to end. Its calls are taken from the upstream
source rather than guessed, but the first run should be checked against a handful of
pairs from `../dataset/pairs.csv` before any number from it is used. That check is
also what will confirm the checkpoint loads from a fresh clone.

## Training

`upstream/examples/train.py` is the upstream training entry point and
`upstream/data/dataset.csv` its dataset. Neither is wired to `../dataset/pairs.csv`
yet.

When it is, the split matters more than it looks: an earlier audit found ML
classifiers on this data were largely memorizing per-photocatalyst λ fingerprints. A
random split lets the same photocatalyst appear on both sides and reports that
memorization as accuracy. Leave-one-photocatalyst-out is the honest measure and is
what the theory benchmark already uses, so it is also the only split that makes the
model-versus-theory comparison in the paper mean anything.

## A note on repository size

`upstream/` is about 111 MB, most of it the 69 MB checkpoint and ~35 MB of Ketcher
editor assets for the web application. That is committed directly rather than through
Git LFS, which works — GitHub's hard limit is 100 MB per file and the checkpoint is
under it — but it will produce a size warning on push. If the checkpoint grows, or if
more are added, move `*.pt` to LFS before that happens rather than after.
