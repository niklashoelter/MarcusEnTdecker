# ml — models and predictions

```
pyrates/          the model, the energy theory and the web application
predict.py        the prediction API: two SMILES in, a rate constant out
tests/            the package's own test suite
examples/train.py the training entry point
training_data/    dataset.csv, what training runs from
pyproject.toml    pins torch, torch-geometric and rdkit — what `--extra ml` installs
LICENSE-pyrates   MIT, Leon Schlosser
```

## Credit and origin

**The `pyrates` package here is Leon Schlosser's work, not this repository's.** It
comes from [le-schlo/pyrates](https://github.com/le-schlo/pyrates) and is MIT
licensed; the licence travels with it as `LICENSE-pyrates` and must stay with any
redistribution. It lives here as ordinary first-party code — develop against it, fix
it, extend it — but the authorship is his, and anything published from it should say
so.

It is kept in this repository rather than pinned as a remote dependency because the
paper's numbers have to stay reproducible after the library moves on, and because the
deployed web application should be built from a commit rather than from a branch head.
That includes the trained checkpoint the application serves,
`pyrates/webapp/data/model.pt` (69 MB).

The code was taken at these commits:

| | source | commit | date |
|---|---|---|---|
| everything | branch `fastapi-webapp` | `c7d5fec279a7d05f03bd2a715d66d2e52bab9fd8` | 2026-07-31 |
| `training_data/dataset.csv` | branch `main` | `8b1de3c74b05940c0b0c28d64a2482e2154629d1` | 2026-07-31 |
| `examples/train.py` | branch `main` | `8b1de3c74b05940c0b0c28d64a2482e2154629d1` | 2026-07-31 |

`fastapi-webapp` is the base because it is the branch the deployed application is
built from, and it is the only place the trained checkpoint exists. The two files
below it are added back because that branch deletes them and they are what training
runs from.

Two changes were made to the tree as copied, both structural, neither touching any
line of Python:

- **`.gitignore` was deleted.** It listed `data/`, which here would have silently
  excluded both the training dataset and the 69 MB checkpoint at
  `pyrates/webapp/data/model.pt` — the two files most necessary to reproduce anything.
  Upstream committed them before that rule existed, so the contradiction is invisible
  there and only bites on a fresh copy.
- **`data/` was renamed to `training_data/`.** The repository root already has a
  `data/` meaning something else entirely (instrument output), and two directories
  called `data/` one level apart is a trap. Nothing reads the old path: `train.py`
  takes its dataset path as an argument, and the package-data globs in
  `pyproject.toml` refer to `pyrates/webapp/data/`, which is untouched.

### Known issue: the two upstream branches have diverged

`main` and `fastapi-webapp` forked at `1ae1a13` on 2026-03-20 and were never merged.
Nine commits on `main` are absent from the deployed `fastapi-webapp` branch, including
`665fd51 update marcus and included dataset`.

**This does not affect the physics.** `pyrates/energy/` is byte-identical between the
two branches, so Marcus and Eyring are the same code either way. The differences are
`pyrates/model/inference.py` (+97 lines on the webapp branch), two plotting modules,
and packaging files (`pyproject.toml`, `LICENSE`, `README.md`).

It should still be reconciled before publication. "Which branch produced the published
numbers" is a question a reviewer can reasonably ask, and the honest answer today is
"the deployed one, which is missing nine commits from main".

## Installing

```
uv sync --extra ml
```

This installs `ml/` in editable mode from `ml/pyproject.toml`, which is what pins
torch, torch-geometric and rdkit. It is deliberately not part of the default install:
the data pipeline needs none of it, and `uv run reproduce.py` should not drag in a
multi-gigabyte dependency tree.

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
pyrates prediction.predict(smiles)    graph -> E_T, lambda_S_T, lambda_T_S
pyrates marcus_theory                 properties -> dG and the activation barrier
Balzani rate expression               barrier -> k, capped at the diffusion limit
```

Only the last step is written here, because `pyrates` stops at the barrier. Keeping
the barrier in `pyrates` means the web application, the paper and this API compute it
with the same code.

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

## Quencher predictions, and what they exposed

`predict_quenchers.py` runs the checkpoint over the 48 screened quenchers and writes
`predictions_quenchers.csv`. Because those 48 also have independently computed DFT
triplet energies, it doubles as the check on whether the model loads and runs correctly.

```
uv sync --extra ml
uv run --extra ml python ml/predict_quenchers.py
```

Against DFT, over all 48 (kcal/mol):

| | |
|---|---:|
| mean signed error | +1.56 |
| MAE | 4.32 |
| RMSE | 9.16 |
| max abs error | **32.76** |
| within 2σ of the model's own uncertainty | 40/48 |

The RMSE is twice the MAE, so the error is concentrated rather than spread. The five
worst are:

| | compound | ml | dft | error | λ_S→T (DFT) |
|---|---|---:|---:|---:|---:|
| Q17 | (BnSe)₂ | 79.9 | 47.1 | **+32.8** | 36.8 |
| Q10 | diphenyl disulfide | 72.2 | 40.3 | **+31.9** | 77.6 |
| Q45 | vinyl selenoxide | 77.2 | 48.3 | **+28.9** | 104.8 |
| Q11 | dibenzyl disulfide | 75.6 | 56.6 | **+19.0** | 45.5 |
| Q44 | 2-acetylpyridine | 65.6 | 82.5 | −16.9 | 11.4 |

**Four of the five are dichalcogenides**, and the likely explanation is that the DFT is
wrong there rather than the model. The triplet state of an S–S or Se–Se compound is
dissociative: an adiabatic triplet optimisation relaxes toward two chalcogenyl radicals,
so the "adiabatic triplet energy" it returns is closer to a homolysis energy than to an
excitation energy. Three things point the same way:

1. **λ_S→T = 104.8 kcal/mol for Q45, against its own triplet energy of 48.3.** A
   reorganization energy twice the excitation energy does not describe a bound excited
   state — the triplet geometry is nowhere near the singlet.
2. **Q10 and Q11 are also the only two quenchers with no reduced-state jobs at all**, so
   this family was already giving the DFT pipeline trouble.
3. Selenium is *in* the model's element vocabulary (`atomic_num` includes 34), so these
   are not silently rejected inputs falling back on garbage features.

The discriminator is not simply a large λ. Styrene, α-methylstyrene, F-styrene and
butyl methacrylate all have λ_S→T between 85 and 88 — C=C twisting in the triplet — and
the model reproduces their DFT triplet energies to about 1 kcal/mol. It is the
dichalcogenides specifically.

**This is not a modelling problem to fix here; it is four quencher entries to check.**
Q10, Q11, Q17 and Q45 carry 27 pairs each, 108 of 1296. Their DFT E_T feeds ΔG_EnT
directly and their λ feeds the Marcus barrier, where a λ of 105 kcal/mol flattens the
barrier for every pair it touches. Whether their triplets are genuinely dissociative
should be settled by looking at the optimised triplet geometries before any of those
values is used.

## predict.py is still not run end to end

`predict_quenchers.py` above exercises the model and the checkpoint, so those are now
confirmed working. `predict.py` itself — the pair-level rate API — has still not been run:
it needs a photocatalyst as well, and the photocatalyst DFT is not in the repository. Its
calls are taken from the `pyrates`
source rather than guessed, but the first run should be checked against a handful of
pairs from `../dataset/pairs.csv` before any number from it is used. That check is
also what will confirm the checkpoint loads from a fresh clone.

## Training

`examples/train.py` is the training entry point and `training_data/dataset.csv` its
dataset. Neither is wired to `../dataset/pairs.csv` yet.

When it is, the split matters more than it looks: an earlier audit found ML
classifiers on this data were largely memorizing per-photocatalyst λ fingerprints. A
random split lets the same photocatalyst appear on both sides and reports that
memorization as accuracy. Leave-one-photocatalyst-out is the honest measure and is
what the theory benchmark already uses, so it is also the only split that makes the
model-versus-theory comparison in the paper mean anything.

## A note on repository size

`ml/` is about 111 MB: the 69 MB checkpoint, ~33 MB of Ketcher editor assets for the
web application, and the 8.7 MB training dataset. That is committed directly rather than
through Git LFS, which works — GitHub's hard limit is 100 MB per file and the
checkpoint is under it — but it will produce a size warning on push. If the checkpoint
grows, or if more are added, move `*.pt` to LFS before that happens rather than after.
