# Triplet energy transfer kinetics — data, benchmarks and models

Companion repository for *A Hybrid Machine-Learning and Marcus Theory Approach for
Predicting Triplet Energy Transfer Kinetics*: experimental and computational data with
their processing scripts, the pair-level dataset, the theory benchmarks, and the
machine-learning models.

## The screen

Every photocatalyst against every quencher: automated Stern–Volmer quenching in
acetonitrile on an Opentrons Flex and a BMG VantaStar plate reader, in 96-well
glass-coated plates with six-point dilution series.

## Layout

```
data/           instrument output and DFT job files, the scripts that process them, and
                the pair-level dataset built from them
benchmark/      feasibility criteria scored against the screen (adiabatic, DvTE and the
                two-gate Marcus criterion), rate predictions, and the literature benchmark
ml/             the `ml` package: the graph MDN, its training and inference, the trained
                checkpoint, and the prediction API and `ent-ml` command line on top
webapp/         the `webapp` package: EnTdecker, the FastAPI application that serves `ml`
                in a browser
```

Each directory has a README; `data/` has one for all of its subfolders.

`data/` and `benchmark/` are scripts. `ml/` and `webapp/` are the two installable
packages, built from the root `pyproject.toml`:

```
uv sync --extra ml         # the models, the prediction API and the ent-ml CLI
uv sync --extra webapp     # the same, plus what serving EnTdecker needs
```

Dependencies run one way:

```
data  ->  benchmark
ml    ->  webapp
```

`benchmark/` reads only what `data/` writes (and the training set and splits in
`ml/data/`). The packages never read anything under `data/` or `benchmark/`, and `ml`
never imports `webapp`; each reads its own resources (`ml/data/model_final.pt`,
`webapp/data/`) with `importlib.resources`. `webapp/tests/test_no_reverse_dependency.py`
enforces this. Three scripts call the `ml` package: `quencher_ml.py` and
`quencher_ml_timing.py` in `data/computational_data_scripts/`, which write
`quencher_properties_ml.xlsx` and `quencher_timings_ml.csv`, and
`benchmark/literature/predict.py`, which writes the predictions of the literature benchmark.

Inside `data/`, the processed folders hold one workbook per compound class and side, and
the Stern–Volmer tables; `build_exp_dataset.py` turns the experimental ones into one row
per pair. `build_exp_dataset_ent.py` keeps the pairs on which quenching can be attributed
to energy transfer and joins the calculated and predicted properties and driving forces
onto them; that table is what `benchmark/` reads:

```
data/experimental_data_processed/    photocatalyst_properties_exp.xlsx
                                     quencher_properties_exp.xlsx
                                     stern_volmer_intensities.csv, stern_volmer_fits.csv
data/computational_data_processed/   photocatalyst_properties_calc.xlsx
                                     quencher_properties_calc.xlsx
                                     quencher_properties_ml.xlsx
                                     quencher_timings.csv, quencher_timings_ml.csv
                                     quencher_geometry_S0_T1.csv,
                                     reorganization_modes.csv   (read by
                                     benchmark/further_analysis/)
                                     molecular_extent.csv
data/                                exp_dataset.csv, exp_dataset.xlsx   (every pair,
                                     measured values and labels only)
data/                                exp_dataset_ent.csv, exp_dataset_ent.xlsx   (all
                                     non-quenching pairs, and the quenching pairs with
                                     electron transfer ruled out; input to benchmark/)
```

Each workbook has one row per compound and its columns in blocks, with a `Column key`
sheet naming the table each column was copied from. Those intermediate tables (fitted
lifetimes, extracted spectra, redox potentials, DFT energies) are in the same processed
folders.

## Reproducing

```
uv run reproduce.py            # the data, benchmark, further_analysis and webapp stages, in dependency order
uv run reproduce.py data       # just one stage
```

Each script declares its dependencies in a PEP 723 header and runs by itself, without
arguments, from any working directory:

```
uv run data/experimental_data_scripts/stern_volmer_fits.py
```

Exceptions: the three scripts that use the `ml` package (above) run with
`uv run --extra ml python <script>`; the comparison scripts of the literature benchmark
are described in `benchmark/README.md`; the model training in `ml/README.md`.

### Not in this repository

`data/computational_data_raw/` (ORCA and Milo job files) is archived to Zenodo. The
tables extracted from it and the workbooks built from them are committed; the archive is
needed only to re-derive the extracted tables.

## Predicting from structure

```python
import dataclasses

from ml.predict import RatePredictor, photocatalyst_properties

photocatalyst = photocatalyst_properties(
    "data/computational_data_processed/photocatalyst_properties_calc.xlsx", 1)
photocatalyst = dataclasses.replace(photocatalyst, tau0_ns=TAU0_NS)   # two_gate needs tau0
result = RatePredictor().predict(photocatalyst, quencher=Q_SMILES)
result.k_ent, result.dG_barrier_kcal, result.uncertainty, result.two_gate
```

Photocatalyst properties (Ir and Ru complexes) come from DFT; the quencher is predicted
from its SMILES. `uv sync --extra ml` first. Command line: `uv run ent-ml --help`;
browser: EnTdecker, in `webapp/`.

## The models, and who wrote them

The model code is not this repository's work. It is
[pyrates](https://github.com/le-schlo/pyrates), by **Leon Schlosser**, carried here as
the `ml` package together with his training scripts. The checkpoint the web application
serves, `ml/data/model_final.pt`, was trained in this repository with that code;
`ml/README.md` describes how, and records the commits the code came from.
