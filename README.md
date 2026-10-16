# MarcusEnTdecker

[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/paper-to%20follow-lightgrey)](#publication)
[![Zenodo](https://img.shields.io/badge/Zenodo-data%20to%20follow-lightgrey?logo=zenodo&logoColor=white)](#publication)
[![ioChem-BD](https://img.shields.io/badge/ioChem--BD-DFT%20to%20follow-lightgrey)](#publication)
[![Web app](https://img.shields.io/badge/web%20app-EnTdecker-2ea44f)](https://entdecker.uni-muenster.de)

This repository accompanies our paper *A Hybrid Machine Learning and Marcus Theory Approach
for Triplet Energy Transfer* (citation to follow).

Its core is an ML framework that predicts, from a SMILES string, the adiabatic triplet energy
<i>E</i><sub>T</sub>, the reorganization energies <i>λ</i><sub>S0-T1</sub> and <i>λ</i><sub>T1-S0</sub>, and the triplet spin populations of a
molecule. Each prediction comes with an uncertainty. Combined with Marcus theory, these
predictions give the driving force and the barrier of triplet energy transfer from a
photocatalyst.

## Installation

Requires Python ≥ 3.12.

```bash
git clone https://github.com/GloriusGroup/MarcusEnTdecker.git
cd MarcusEnTdecker
```

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync --extra ml          # ML models, Python API and the ent-ml command line
uv sync --extra webapp      # the same, plus the web application
```

With pip:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[ml]"      # or ".[webapp]"
```

The trained model (`ml/data/model_final.pt`) is part of the repository.

## Usage

Command line:

```bash
uv run ent-ml predict "C=Cc1ccccc1"                  # triplet and reorganization energies, uncertainties
uv run ent-ml predict --file smiles.txt --format csv --out predictions.csv
uv run ent-ml rate --quencher "C=Cc1ccccc1" \
    --photocatalysts data/computational_data_processed/photocatalyst_properties_calc.xlsx \
    --photocatalyst 1                                # driving force, barrier, rate constant
uv run ent-ml --help
```

Python:

```python
from ml.prediction import predict
from ml.predict import RatePredictor, photocatalyst_properties

predict(["C=Cc1ccccc1"])        # DataFrame: triplet and reorganization energies, uncertainties, spin populations

pc = photocatalyst_properties(
    "data/computational_data_processed/photocatalyst_properties_calc.xlsx", 1)   # PC1
result = RatePredictor().predict(pc, quencher="C=Cc1ccccc1")
result.dG_kcal, result.dG_barrier_kcal, result.k_ent
```

The photocatalyst properties (<i>E</i><sub>T</sub>, <i>λ</i>) come from DFT. `ml/README.md` describes the model, its
training and the full API.

## Web application

EnTdecker serves the model in the browser and computes the EnT barrier for any molecule
against the 35 photocatalysts of our library. It is hosted at
https://entdecker.uni-muenster.de. To run it locally:

```bash
uv sync --extra webapp
uv run uvicorn webapp.app:app --port 5000           # then open http://127.0.0.1:5000
```

## Reproducing the paper

The data, the processing scripts, the benchmarks and the analyses behind every number in the
paper are in `data/` and `benchmark/`; `data/README.md` and `benchmark/README.md` describe
them. The raw data (`data/experimental_data_raw/`, `data/computational_data_raw/`) are not
part of this repository. They are archived on Zenodo together with the version of this
repository used for the paper (DOI to follow).

```bash
uv run reproduce.py              # all stages: data, benchmark, further_analysis, webapp
uv run reproduce.py benchmark    # one stage
```

Without the raw data, `reproduce.py` skips the extraction scripts and rebuilds everything
else from the committed tables. To also rerun the extraction, unpack the Zenodo archive and
place its two raw folders in `data/`.

Tests: `uv run --extra webapp --group dev pytest`.

## Publication

*A Hybrid Machine Learning and Marcus Theory Approach for Triplet Energy Transfer*,
in preparation. DOI and citation to follow.

- Paper: DOI to follow
- Data and archived code: Zenodo, DOI to follow
- DFT calculations: ioChem-BD, link to follow
- Web application: https://entdecker.uni-muenster.de

If you use this code or data, please cite the paper.

## Credits

The model code is [pyrates](https://github.com/le-schlo/pyrates) by Leon Schlosser, carried
here as the `ml` package; `ml/README.md` records its origin.
