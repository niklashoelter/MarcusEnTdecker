# ml — the model, and the predictions built on it

```
prediction.py       model wrapper: SMILES in, E_T and both reorganization energies out
predict.py          prediction API: a pair in, a rate constant out
cli.py              the `ent-ml` command
model/              the graph MDN, its loss and its inference
energy/             Marcus, Eyring and the two-gate verdict
utils/              graph construction and the molecule dataset
training/           offline only, not in the wheel
  splits.py         train/validation/test splits in data/splits/
  hyperparameters.py   the hyperparameters, Optuna trial 55
  train.py          trains on a split, or on all data
  test_predictions.py   a checkpoint's predictions for the test set of its split
  evaluate.py       metrics from the test predictions
  hyperparameter_optimization.py   the Optuna search
  loop.py           one training or validation epoch, and the loss
data/
  model_final.pt              the deployed checkpoint, a copy of results/full/model.pt
  training_dataset_full.csv   the training dataset
  splits/                     random_<seed>.csv and oos_styryl.csv
results/            what training/ writes:
  random_<seed>/, oos_styryl/   model.pt, test_predictions_graph.csv, test_predictions_node.csv
  full/model.pt                 the model trained on all data
  metrics.csv, metrics_random_splits.csv
  hyperparameter_optimization/optuna_results.csv
tests/              the package's test suite
```

## Installing

```
uv sync --extra ml
```

## Inputs

The prediction interface loads `ml/data/model_final.pt` via `importlib.resources`.
Offline training and evaluation use the datasets, splits and checkpoints under `ml/`.

The photocatalyst table is an argument: `photocatalyst_properties(source, pc_id)` takes a
path to an .xlsx (headers on row 2) or .csv, a pandas DataFrame, or a mapping, with one row
per photocatalyst carrying `new_id`, `E_T / kcal mol-1`, `lambda_S0_T1 / kcal mol-1` and
`lambda_T1_S0 / kcal mol-1`, and optionally `tau0 / ns`, the excited-state lifetime.

## Python API

```python
from ml.prediction import predict

predict([smiles, ...])          # DataFrame: triplet_energy_adiabatic, lamda_S_T, lamda_T_S,
                                # each with _uncertainty and its mixture, and spin_map
```

```python
from ml.predict import RatePredictor, photocatalyst_properties

photocatalyst = photocatalyst_properties("photocatalyst_properties_calc.xlsx", 1)
result = RatePredictor().predict(photocatalyst, quencher=Q_SMILES)

result.dG_kcal          # driving force
result.dG_barrier_kcal  # Marcus activation barrier
result.k_ent            # rate constant, M-1 s-1
result.t_pc_kcal        # lifetime barrier t_PC from tau0; None without tau0
result.two_gate         # dG <= 0 and barrier <= t_PC; None without tau0
result.uncertainty      # from the mixture density network
```

The two-gate verdict (`ml.energy.two_gate`) needs `tau0_ns`, from a `tau0 / ns` column or
set with `dataclasses.replace(photocatalyst, tau0_ns=...)`. t_PC is the barrier at which
k·τ₀ reaches K_SV,crit in the exergonic limit; the constants are in `energy/two_gate.py`.

`predict_many` takes an iterable of pairs and predicts each distinct SMILES once.

## Command line

```
ent-ml predict SMILES [SMILES ...]        properties of each structure
ent-ml predict --file quenchers.smi       one SMILES per line
       --format table|csv|json  --out predictions.csv

ent-ml rate --quencher SMILES --photocatalysts photocatalysts.xlsx
       [--photocatalyst 1] [--tau0 NS]     with tau0, also t_pc_kcal and two_gate

ent-ml info                               the checkpoint and its element vocabulary
```

An invalid SMILES or a missing file exits non-zero with a one-line message.

## Element vocabulary

**H, B, C, N, O, F, Si, P, S, Cl, Se, Br, Sn, I.** Encoding Ir or Ru raises
`UnknownAtomError`, so a metal-complex photocatalyst is passed as DFT properties:

| partner | comes from | how it is passed |
|---|---|---|
| quencher | the model | a SMILES |
| photocatalyst | DFT | `photocatalyst_properties(table, pc_id)` |

Either argument accepts either form. `uncertainty` covers only the model-predicted
partners, named in `uncertainty_covers`.

## Rate expression

```
ml.prediction.predict(smiles)    graph -> E_T, lambda_S_T, lambda_T_S
ml.energy.marcus_ent             properties -> dG and the activation barrier
ml.energy.two_gate               tau0 -> t_PC and the two-gate verdict; k_-d/ν
encounter-complex rate           barrier -> k
```

`k_EnT = k_diff / (1 + exp(dG/RT) + (k_-d/ν) exp(dG‡/RT))`, with
`k_-d/ν = (k_diff/K_A)/(k_BT/h)` and K_A the Eigen encounter constant at the contact
distance of the two-gate verdict (`energy/two_gate.py`, `escape_over_frequency`). The
Eyring rate of the barrier is returned as `k_eyring`.

## Training

`ml/data/training_dataset_full.csv` is the training dataset. The model is validated on five
random splits and one out-of-sample split, with the same settings and hyperparameters;
`model_final.pt` is trained on all data.

```
uv run --extra ml python -m ml.training.splits
uv run --extra ml python -m ml.training.train random_42 random_101 random_143 random_404 random_666 --device cuda:0
uv run --extra ml python -m ml.training.train oos_styryl --device cuda:0
uv run --extra ml python -m ml.training.train full --device cuda:0
uv run --extra ml python -m ml.training.evaluate
cp ml/results/full/model.pt ml/data/model_final.pt
```

`train.py` writes `ml/results/<split>/model.pt` (lowest validation loss) and its test
predictions; with `full` it trains on every molecule for the median best epoch of the five
random-split runs and keeps the last epoch. An existing `model.pt` is not overwritten (only
the test predictions are rewritten); move it away to retrain. `test_predictions.py` rewrites
the test predictions of a split. `evaluate.py` writes `ml/results/metrics.csv` (per split and
target) and `metrics_random_splits.csv` (mean and SD over the random splits).

Test predictions: `test_predictions_graph.csv`, one row per test molecule (`idx`, `smiles`,
and `<target>_target`, `_pred`, `_uncertainty` per graph target); `test_predictions_node.csv`,
one row per atom (`idx`, `atom`, and the same three columns for the spin population). `idx`
is the molecule's `idx` in the training dataset.

Split files: one row per molecule in dataset order (`idx`, `split`); `read_split(name)`
returns positional train, validation and test indices.

| split | test set | train : val |
|---|---|---|
| `random_<seed>`, seeds 42, 101, 143, 404, 666 | 10 %, by `train_test_split` | 80 : 10 |
| `oos_styryl` | every molecule matching the SMARTS `c1ccc(cc1)C=C` | 80 : 20 of the rest, shuffled with seed 42 |

Both procedures are Leon's own; the styryl split is the out-of-sample split of his thesis.

`training/hyperparameter_optimization.py` is the Optuna search; its results are in
`results/hyperparameter_optimization/`, and `hyperparameters.py` holds trial 55 (lowest
validation loss).

## Credit and origin

The model code is Leon Schlosser's work, from
[le-schlo/pyrates](https://github.com/le-schlo/pyrates), renamed to `ml` and arranged
into runtime and offline training modules in this repository.

The code was taken at these commits:

| | source | commit | date |
|---|---|---|---|
| the model code | branch `fastapi-webapp` | `c7d5fec279a7d05f03bd2a715d66d2e52bab9fd8` | 2026-07-31 |
| `data/training_dataset_full.csv` | branch `main` | `8b1de3c74b05940c0b0c28d64a2482e2154629d1` | 2026-07-31 |

Upstream's `data/dataset.csv` is carried here as
`data/training_dataset_full.csv`, byte-identical to upstream.

The runtime code was reorganized into `model/` and `utils/`, with explicit checkpoint
selection and package-resource loading. The offline training, evaluation and
hyperparameter-optimization scripts were adapted from Leon Schlosser's research code
to use this package's dataset and splits.

The models in `results/` and `model_final.pt` were trained in this repository with
trial 55.
`results/hyperparameter_optimization/optuna_results.csv` is the search's output, unchanged.
