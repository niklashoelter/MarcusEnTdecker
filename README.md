# Triplet energy transfer kinetics — data, benchmarks and models

Companion repository for *A Hybrid Machine-Learning and Marcus Theory Approach for
Predicting Triplet Energy Transfer Kinetics*.

It holds everything needed to regenerate the paper's numbers from instrument output:
the experimental and computational data with the scripts that process them and the
pair-level dataset built from them, the physical-organic theory benchmarks, and the
training and evaluation of the machine-learning models.

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

Each directory's README says what is in it, what produced it, and what is known to be
wrong with it; `data/` has one README for all of its subfolders.

`data/` and `benchmark/` are scripts. `ml/` and `webapp/` are the two installable
packages, and both are built from the single root `pyproject.toml`:

```
uv sync --extra ml         # the models, the prediction API and the ent-ml CLI
uv sync --extra webapp     # the same, plus what serving EnTdecker needs
```

Dependencies run one way:

```
data  ->  benchmark
ml    ->  webapp
```

`benchmark/` reads only what `data/` writes, so any number in the paper can be traced to
the instrument file it came from. The packages never read anything under `data/` or
`benchmark/`, and `ml` never imports `webapp`; each carries its own resources —
`ml/data/model_final.pt`, `webapp/data/` — and reads them with `importlib.resources`.
`webapp/tests/test_no_reverse_dependency.py` keeps that from regressing. The two sides
meet in three scripts that call the `ml` package: `quencher_ml.py` and
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
data/                                exp_dataset.csv, exp_dataset.xlsx   (every pair,
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
uv run reproduce.py            # the data, benchmark and further_analysis stages, in dependency order
uv run reproduce.py data       # just one stage
```

Individual scripts are standalone and declare their own dependencies in a PEP 723
header, so any one of them also runs by itself:

```
uv run data/experimental_data_scripts/stern_volmer_fits.py
```

No install step, no environment to activate, no arguments. Paths are resolved relative
to each script's own location, so the working directory does not matter. The exceptions
are the three scripts that use the `ml` package (above), which run with
`uv run --extra ml python <script>`, the comparison scripts of the literature benchmark,
which are not a stage of `reproduce.py` and which `benchmark/README.md` describes, and the
model training, which `ml/README.md` describes.

### The one thing that is not in this repository

`data/computational_data_raw/` is about 11 GB of ORCA and Milo job files. It is
archived to Zenodo rather than versioned here. The tables extracted from it are
committed — in the same folders, next to where the jobs go — and so are the workbooks
built from them, so the repository is fully usable without it. You only need it to
re-derive the extracted tables from scratch.

## Predicting from structure

```python
from ml.predict import RatePredictor, photocatalyst_properties

photocatalyst = photocatalyst_properties(
    "data/computational_data_processed/photocatalyst_properties_calc.xlsx", 1)
result = RatePredictor().predict(photocatalyst, quencher=Q_SMILES)
result.k_ent, result.dG_barrier_kcal, result.uncertainty, result.two_gate
```

The photocatalysts are Ir and Ru complexes, which the model cannot read, so their
properties come from DFT; the quencher is predicted from its SMILES. `uv sync --extra ml`
first. The same calls are also a command line — see `uv run ent-ml --help` — and for a
browser instead, that is what EnTdecker is, in `webapp/`.

## The models, and who wrote them

The model code is not this repository's work. It is
[pyrates](https://github.com/le-schlo/pyrates), by **Leon Schlosser**, carried here as
the `ml` package together with his training scripts. The checkpoint the web application
serves, `ml/data/model_final.pt`, was trained in this repository with that code;
`ml/README.md` describes how, and records the commits the code came from.

It is carried in this repository rather than pinned as a remote dependency because the
paper's numbers have to stay reproducible after the library moves on, and because the
deployed web application should be built from a commit rather than from a branch head.
