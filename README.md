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
ml/             the pyrates models and the prediction API
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
| `benchmark/` | **ported and running** — five theories scored; open items in its README |
| `ml/` | **models in place**, checkpoint included; `predict.py` written but not yet run |
| `webapp/` | **deployable** — builds from `ml/` instead of cloning a branch |
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

## The models, and who wrote them

The models are not this repository's work. They are
[pyrates](https://github.com/le-schlo/pyrates), by **Leon Schlosser**, MIT licensed,
and the package lives at `ml/pyrates/` — including the trained checkpoint the web
application serves. His licence is kept alongside it as `ml/LICENSE-pyrates` and must
travel with any redistribution.

It is carried in this repository rather than pinned as a remote dependency because the
paper's numbers have to stay reproducible after the library moves on, and because the
deployed web application should be built from a commit rather than from a branch head.
`ml/README.md` records the exact commits the code came from, the two structural changes
made to the tree, and one open issue — the two source branches diverged and were never
merged, which does not affect the physics but should be reconciled before publication.

## Citing

See `CITATION.cff`. Code is MIT (`LICENSE`); data is CC BY 4.0 (`LICENSE-DATA`).
