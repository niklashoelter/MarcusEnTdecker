# Triplet energy transfer kinetics — data, benchmarks and models

Companion repository for *A Hybrid Machine-Learning and Marcus Theory Approach for
Predicting Triplet Energy Transfer Kinetics*.

It holds everything needed to regenerate the paper's numbers from instrument output:
the experimental and computational data with the scripts that process them, the
assembled pair-level dataset, the physical-organic theory benchmarks, and the training
and evaluation of the machine-learning models.

## The screen

27 photocatalysts × 48 quenchers = **1296 pairs**, complete. Automated Stern–Volmer
quenching in acetonitrile on an Opentrons Flex and a BMG VantaStar plate reader;
96-well glass-coated plates, six-point dilution series.

## Layout

```
data/           instrument output and DFT job files, and the scripts that process them
dataset/        the pair-level table -- where the processed data becomes one row per pair
benchmark/      physical-organic theory as classifiers: adiabatic, Marcus, Rehm-Weller, DvTE
ml/             the models, vendored from pyrates, and the prediction API
figures/        every figure in the paper, regenerated from processed data
paper/          tables destined for the manuscript and SI, generated rather than typed
webapp/         shipping EnTdecker: image definition and Kubernetes manifests
```

Dependencies run strictly one way:

```
data  ->  dataset  ->  { benchmark, ml }  ->  figures, paper
```

Nothing reaches sideways or backwards. A script reads only from directories to its
left, so any number in the paper can be traced to the instrument file it came from by
following that chain in one direction.

## Reproducing

```
uv run reproduce.py            # everything, in dependency order
uv run reproduce.py data       # just one stage
```

Individual scripts are standalone and declare their own dependencies in a PEP 723
header, so any one of them also runs by itself:

```
uv run data/experimental_data_scripts/stern_volmer_fits.py
```

No install step, no environment to activate, no arguments. Paths are resolved relative
to each script's own location, so the working directory does not matter.

### The one thing that is not in this repository

`data/computational_data_raw/` is about 11 GB of ORCA and Milo job files. It is
archived to Zenodo rather than versioned here, and everything downstream of it is
already committed as processed CSVs — so the repository is fully usable without it.
You only need it to re-derive the computational tables from scratch. See
`data/computational_data_raw/README.md`.

## State

| stage | state |
|---|---|
| `data/` experimental | **complete and verified** — 5 of 6 scripts written, all round-trip byte-identical |
| `data/` computational | **quenchers complete and verified**; photocatalyst job files not yet available |
| `dataset/` | in progress |
| `benchmark/` | to be ported from `phys_org_theories` |
| `ml/` | **models vendored**, checkpoint included; `predict.py` written but not yet run |
| `webapp/` | **deployable** — builds from the vendored copy instead of cloning a branch |
| `figures/`, `paper/` | not started |

Each directory's own README says what is in it, what produced it, and what is known to
be wrong with it.

## Predicting from structure

```python
from ml.predict import RatePredictor

predictor = RatePredictor()
result = predictor.predict(photocatalyst=PC_SMILES, quencher=Q_SMILES)
result.k_ent, result.dG_barrier_kcal, result.uncertainty, result.quenches
```

`uv sync --extra ml` first. There is no command-line interface; for a browser, that is
what EnTdecker is.

## Relationship to pyrates

The models are not implemented here. They are
[pyrates](https://github.com/le-schlo/pyrates), by Leon Schlosser, MIT licensed, and a
copy of it lives at `ml/upstream/` — including the trained checkpoint the web
application serves. `ml/upstream/PROVENANCE.md` records the exact commits and the one
modification made (deleting a `.gitignore` that would have excluded both the dataset
and the checkpoint).

It is vendored rather than pinned as a remote dependency because this repository has to
keep reproducing the paper's numbers after upstream moves on, and because the deployed
web application should be built from a commit rather than from a branch head. Do not
develop against `ml/upstream/`: fixes belong upstream and return here as a re-vendor.

## Citing

See `CITATION.cff`. Code is MIT (`LICENSE`); data is CC BY 4.0 (`LICENSE-DATA`).
