# MarcusEnTdecker

![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue?logo=python&logoColor=white)
![Paper](https://img.shields.io/badge/paper-link%20to%20follow-lightgrey)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.23505512-blue?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.23505512)
[![ioChem-BD](https://img.shields.io/badge/ioChem--BD-DFT%20data-blue)](https://doi.org/10.19061/iochem-bd-6-716)
[![Web app](https://img.shields.io/badge/web%20app-EnTdecker-2ea44f)](https://entdecker.uni-muenster.de)

This repository accompanies our manuscript:

Hölter, N.; Schlosser, L.; Hümpel, C.; Spierling, L.; Glorius, F.
*A Hybrid Machine Learning and Marcus Theory Approach for Rationalizing and Predicting Triplet Energy Transfer*.
Manuscript in preparation; publication link to follow.

It builds on the original EnTdecker platform:

Schlosser, L.; Rana, D.; Pflüger, P.; Katzenburg, F.; Glorius, F.
*EnTdecker—A Machine Learning-Based Platform for Guiding Substrate Discovery in Energy Transfer Catalysis*.
**J. Am. Chem. Soc. 2024**, *146* (19), 13266–13275.
[DOI: 10.1021/jacs.4c01352](https://doi.org/10.1021/jacs.4c01352).

The ML framework in this repository predicts, from a SMILES string, the adiabatic triplet energy
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

The data-processing pipeline and benchmark analyses are in `data/` and `benchmark/`;
`data/README.md` and `benchmark/README.md` describe them. Some SI statistics and
figure-specific exports were generated outside this repository.

The Zenodo archive includes the code, models, processed data and both raw-data folders:
`data/experimental_data_raw/` and `data/computational_data_raw/`. The GitHub checkout
excludes these raw folders; download them from the Zenodo deposit to run the full pipeline.

```bash
uv run reproduce.py              # all stages: data, benchmark, further_analysis, webapp
uv run reproduce.py benchmark    # one stage
```

After extracting the Zenodo archive, run the commands above from its repository folder.
The raw inputs are already in `data/`. In a GitHub checkout without the raw inputs,
`reproduce.py` skips extraction and rebuilds the remaining stages from the supplied tables.

Tests: `uv run --extra webapp --group dev pytest`.

## Publication

- Current manuscript: publication link to follow
- Data and archived code: [Zenodo](https://doi.org/10.5281/zenodo.23505512)
- DFT calculations: [ioChem-BD](https://doi.org/10.19061/iochem-bd-6-716)
- Web application: [EnTdecker](https://entdecker.uni-muenster.de)

If you use this code or data, please cite our manuscript and the original EnTdecker
publication listed above.

## Credits

The model code is [pyrates](https://github.com/le-schlo/pyrates) by Leon Schlosser, carried
here as the `ml` package; `ml/README.md` records its origin.
