# Triplet energy transfer kinetics — data, benchmarks and models

Companion repository for *A Hybrid Machine-Learning and Marcus Theory Approach for
Predicting Triplet Energy Transfer Kinetics*.

It holds everything needed to regenerate the paper's numbers from instrument output:
the experimental and computational data with the scripts that process them and the
pair-level dataset built from them, the physical-organic theory benchmarks, and the
training and evaluation of the machine-learning models.

## The screen

27 photocatalysts × 48 quenchers = **1296 pairs**, complete. Automated Stern–Volmer
quenching in acetonitrile on an Opentrons Flex and a BMG VantaStar plate reader;
96-well glass-coated plates, six-point dilution series.

## Layout

```
data/           instrument output and DFT job files, the scripts that process them, and
                the pair-level dataset built from them
benchmark/      physical-organic theory as classifiers: adiabatic, Marcus, Rehm-Weller, DvTE
ml/             the pyrates models and the prediction API
figures/        every figure in the paper, regenerated from processed data
paper/          tables destined for the manuscript and SI, generated rather than typed
webapp/         shipping EnTdecker: image definition and Kubernetes manifests
```

Dependencies run strictly one way:

```
data  ->  { benchmark, ml }  ->  figures, paper
```

Nothing reaches sideways or backwards. A script reads only from directories to its
left, so any number in the paper can be traced to the instrument file it came from by
following that chain in one direction.

Inside `data/`, the processed folders hold one workbook per compound class and side, and
the Stern–Volmer tables; `build_exp_dataset.py` joins them into one row per pair:

```
data/experimental_data_processed/    photocatalyst_properties_exp.xlsx
                                     quencher_properties_exp.xlsx
                                     stern_volmer_intensities.csv, stern_volmer_fits.csv
data/computational_data_processed/   photocatalyst_properties_calc.xlsx
                                     quencher_properties_calc.xlsx
data/                                exp_dataset.csv, exp_dataset.xlsx   (1296 pairs)
```

Each workbook has one row per compound and its columns in blocks, with a `Column key`
sheet naming the table each column was copied from. Those tables — fitted lifetimes,
extracted spectra, CV traces, conformer energies — are intermediates, and sit in the raw
folder beside the instrument files or jobs they were derived from.

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
archived to Zenodo rather than versioned here. The tables extracted from it are
committed — in the same folders, next to where the jobs go — and so are the workbooks
built from them, so the repository is fully usable without it. You only need it to
re-derive the extracted tables from scratch.

## State

| stage                 | state                                                                                                                                      |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `data/` experimental  | **complete and verified** — every table rebuilds byte-identically from the instrument files                                                |
| `data/` computational | **complete and verified** — 48 quenchers and 27 photocatalysts; reproduces the published tables except four stale photocatalyst E_T values |
| `data/exp_dataset`    | **1296 pairs**, PC1–PC27 by new id; the ground truth is settled on K_SV, both readings carried                                             |
| `benchmark/`          | **ported and running** — five theories scored under both label regimes on 1248 pairs (PC22 not scored); open items in its README           |
| `ml/`                 | **models in place**, checkpoint included; `predict.py` runs end to end, but its λ sweep predates the label change                          |
| `webapp/`             | **deployable** — builds from `ml/` instead of cloning a branch                                                                             |
| `figures/`, `paper/`  | not started                                                                                                                                |

Each directory's README says what is in it, what produced it, and what is known to be
wrong with it; `data/` has one README for all of its subfolders.

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
