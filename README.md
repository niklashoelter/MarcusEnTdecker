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
benchmark/      physical-organic criteria scored against the screen: adiabatic, DvTE,
                Marcus and Rehm-Weller, as classifiers and as rate predictions
ml/             the `ml` package: the graph MDN, its training and inference, the trained
                checkpoint, and the prediction API and `ent-ml` command line on top
webapp/         the `webapp` package: EnTdecker, the FastAPI application that serves `ml`
                in a browser
```

`data/` and `benchmark/` are scripts. `ml/` and `webapp/` are the two installable
packages, and both are built from the single root `pyproject.toml`:

```
uv sync --extra ml         # the models, the prediction API and the ent-ml CLI
uv sync --extra webapp     # the same, plus what serving EnTdecker needs
```

Dependencies run strictly one way:

```
data  ->  { benchmark, ml  ->  webapp }
```

Nothing reaches sideways or backwards. A script reads only from directories to its
left, so any number in the paper can be traced to the instrument file it came from by
following that chain in one direction. The packages are held to the same rule from the
other side: neither `ml` nor `webapp` may read anything under `data/` or `benchmark/`,
and `ml` may not import `webapp`. Each package carries its own resources instead —
`ml/data/model_final.pt`, `webapp/data/` — and reads them with `importlib.resources`.
`webapp/tests/test_no_reverse_dependency.py` is what keeps that from regressing.
Figures and generated tables will be made inside `benchmark/`, from its output.

Inside `data/`, the processed folders hold one workbook per compound class and side, and
the Stern–Volmer tables; `build_exp_dataset.py` turns the experimental ones into one row
per pair. `build_exp_dataset_ent.py` keeps the pairs on which quenching can be attributed
to energy transfer and joins the calculated properties and driving forces onto them;
that table is what `benchmark/` reads:

```
data/experimental_data_processed/    photocatalyst_properties_exp.xlsx
                                     quencher_properties_exp.xlsx
                                     stern_volmer_intensities.csv, stern_volmer_fits.csv
data/computational_data_processed/   photocatalyst_properties_calc.xlsx
                                     quencher_properties_calc.xlsx
data/                                exp_dataset.csv, exp_dataset.xlsx   (1296 pairs,
                                     measured values and labels only)
data/                                exp_dataset_ent.csv, exp_dataset_ent.xlsx   (all
                                     non-quenching pairs, and the quenching pairs with
                                     electron transfer ruled out; input to benchmark/)
```

Each workbook has one row per compound and its columns in blocks, with a `Column key`
sheet naming the table each column was copied from. Those tables — fitted lifetimes,
extracted spectra, redox potentials, conformer energies — are intermediates, and sit in the raw
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
| `data/` computational | **complete** — 48 quenchers and 27 photocatalysts; open recomputations are listed outside the repository |
| `data/exp_dataset`    | **1296 pairs**, PC1–PC27 by new id; twelve experimental columns, nothing calculated; the ground truth is settled on K_SV, both readings carried |
| `benchmark/`          | **in place** — classification, computational cost and further analyses; results only in its `results/` folders, see its README           |
| `ml/`                 | **models in place**, checkpoint included; `predict.py` runs end to end, but its λ sweep predates the label change                          |
| `webapp/`             | **deployable** — serves `webapp.app:app`, built from this repository rather than a cloned branch; `deploy/` is untracked                |

Each directory's README says what is in it, what produced it, and what is known to be
wrong with it; `data/` has one README for all of its subfolders.

## Predicting from structure

```python
from ml.predict import RatePredictor

predictor = RatePredictor()
result = predictor.predict(photocatalyst=PC_SMILES, quencher=Q_SMILES)
result.k_ent, result.dG_barrier_kcal, result.uncertainty, result.quenches
```

`uv sync --extra ml` first. The same calls are also a command line — `ent-ml`, see
`ent-ml --help` — and for a browser instead, that is what EnTdecker is, in `webapp/`.

## The models, and who wrote them

The model code is not this repository's work. It is
[pyrates](https://github.com/le-schlo/pyrates), by **Leon Schlosser**, MIT licensed,
carried here as the `ml` package together with his training scripts. The checkpoint the
web application serves, `ml/data/model_final.pt`, was trained in this repository with that
code; `ml/README.md` describes how. His licence is kept alongside it as
`ml/LICENSE-pyrates` and must travel with any redistribution.

It is carried in this repository rather than pinned as a remote dependency because the
paper's numbers have to stay reproducible after the library moves on, and because the
deployed web application should be built from a commit rather than from a branch head.
`ml/README.md` records the exact commits the code came from, the two structural changes
made to the tree, and one open issue — the two source branches diverged and were never
merged, which does not affect the physics but should be reconciled before publication.

## Citing

See `CITATION.cff`. Code is MIT; data is CC BY 4.0.
