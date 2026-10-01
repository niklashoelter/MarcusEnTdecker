# ml — the model, and the predictions built on it

```
prediction.py       the model wrapper: SMILES in, E_T and both reorganization energies out
predict.py          the prediction API: a pair in, a rate constant out
cli.py              the `ent-ml` command
model/              the graph MDN, its loss and its inference
  checkpoint.py     lets the committed checkpoint unpickle after the package rename
energy/             Marcus and Eyring — the theory the web application and the paper share
utils/              graph construction and the molecule dataset
training/           offline only, left out of the wheel; nothing else may import it
  build_dataset.py  training targets from raw DFT energies
  loop.py           one training or validation epoch, and the loss composition
  metrics.py        spin-population metrics
  splits.py         builds the train/validation/test splits in data/splits/
data/
  model_final.pt              the trained checkpoint, 69 MB, loaded through importlib.resources
  training_dataset_full.csv   the training dataset
  splits/                     random_<seed>.csv and oos_styryl.csv, written by splits.py
tests/              the package's own test suite
LICENSE-pyrates     MIT, Leon Schlosser
```

## Installing

```
uv sync --extra ml
```

It is deliberately optional: the data pipeline and the theory benchmark need none of
torch, torch-geometric or rdkit, and `uv run reproduce.py` should not drag in a
multi-gigabyte dependency tree.

## It reads nothing outside itself

The only file the package opens on its own is its checkpoint, `ml/data/model_final.pt`,
found through `importlib.resources` rather than through a path. It never reads `data/`
or `benchmark/`, and it will work unchanged in a clone that has neither directory.

Anything else it needs is an argument. In particular the photocatalyst table:
`photocatalyst_properties(source, pc_id)` takes a path to an .xlsx or .csv, a pandas
DataFrame, or a mapping, with one row per photocatalyst carrying `new_id`,
`E_T / kcal mol-1`, `lambda_S0_to_T1 / kcal mol-1` and `lambda_T1_to_S0 / kcal mol-1`.
There is no built-in default, because a default would be a claim about where the
caller keeps their data.

## Python API

```python
from ml.prediction import predict

predict([smiles, ...])          # DataFrame: E_T, lambda_S_T, lambda_T_S, uncertainties
```

```python
from ml.predict import RatePredictor, photocatalyst_properties

photocatalyst = photocatalyst_properties("photocatalysts.xlsx", 1)
result = RatePredictor(lambda_scale=0.4).predict(photocatalyst, quencher=Q_SMILES)

result.dG_kcal          # driving force
result.dG_barrier_kcal  # Marcus activation barrier
result.k_ent            # rate constant, M-1 s-1
result.quenches         # k_ent above the detection threshold
result.uncertainty      # from the mixture density network
```

`predict_many` takes an iterable of pairs and runs one forward pass per *structure*,
not per pair, so screening 48 quenchers against one photocatalyst costs 48 passes
rather than 96.

## Command line

`ent-ml` is the same two calls from a shell. Torch is imported only where it is
needed, so `--help` returns immediately.

```
ent-ml predict SMILES [SMILES ...]        properties of each structure
ent-ml predict --file quenchers.smi       one SMILES per line
       --format table|csv|json  --out predictions.csv

ent-ml rate --quencher SMILES --photocatalysts photocatalysts.xlsx
       [--photocatalyst 1] [--lambda-scale 0.4]

ent-ml info                               checkpoint, epoch, val loss, vocabulary
```

An invalid SMILES or a missing file exits non-zero with one line, not a traceback.

## The model cannot see the photocatalysts

Its element vocabulary is **H, B, C, N, O, F, Si, P, S, Cl, Se, Br, Sn, I**. Iridium
(77) and ruthenium (44) are not in it, and encoding one raises `UnknownAtomError`
rather than returning a bad answer. All 27 screened photocatalysts are Ir or Ru
complexes, so none can be passed to the model as a structure. That is a property of
the training set — organic triplet sensitizers — not a defect.

So a pair is predicted from two sources, and the API takes them as two kinds of
argument:

| partner | comes from | how it is passed |
|---|---|---|
| quencher | the model | a SMILES |
| photocatalyst | DFT | `photocatalyst_properties(table, pc_id)` |

Both arguments accept either form, so an organic photocatalyst can still be given as a
SMILES. Passing a metal complex as a SMILES fails loudly, deliberately: a silent
fallback would hide which half of a prediction is a model output.

**A rate from this API is therefore a hybrid**, a calculated triplet energy for one
partner and a predicted one for the other, carrying different errors. `uncertainty`
reports only what the model estimates and `uncertainty_covers` names the partners it
accounts for. It is not a total.

## What `predict.py` composes

Nothing in it reimplements the model or the theory; it is the seam.

```
ml.prediction.predict(smiles)    graph -> E_T, lambda_S_T, lambda_T_S
ml.energy.marcus_ent             properties -> dG and the activation barrier
Balzani rate expression          barrier -> k, capped at the diffusion limit
```

Only the last step is written there, because `ml.energy` stops at the barrier. The
rate is `k_diff / (1 + exp(dG/RT) + 0.25 exp(dG‡/RT))`, which saturates at the
diffusion limit. A bare Eyring rate has no such ceiling and will return 1e14 M⁻¹ s⁻¹
for a strongly exergonic pair, so it is exposed as `k_eyring` for comparison but is
not what `k_ent` reports.

### λ-scaling

`RatePredictor(lambda_scale=...)` scales the predicted reorganization energies before
they reach the barrier. It defaults to 1.0 — raw prediction, nothing silently applied
— but 1.0 is known to over-estimate the barrier. Set it deliberately; the benchmark's
tuned scale factors are in its result files, not here.

## Where it is least trustworthy

The screened quenchers also have DFT triplet energies (`q_E_T_dft` and `q_E_T_ml` in
`data/exp_dataset_ent.csv`), and the difference between the two is concentrated rather
than spread: the largest deviations are dichalcogenides — disulfides and selenium
compounds — predicted well above DFT. The likely explanation is that the DFT is wrong
there: an S–S or Se–Se triplet is dissociative, so an adiabatic triplet optimisation
returns something closer to a homolysis energy, and their DFT reorganization energies
are implausibly large for the same reason. Selenium is in the vocabulary, so these are
not silently rejected inputs. Check those DFT entries before trusting either number for
them.

## Training

`data/training_dataset_full.csv` is the training dataset. There is no training,
hyperparameter-optimisation or evaluation script in this repository yet: upstream does
not publish them. `training/` holds the pieces they build on — the epoch loop in
`loop.py`, the target computation in `build_dataset.py`, the metrics — and the splits.

`training/splits.py` writes the splits to `data/splits/`, one file
per split with one row per molecule in dataset order (`idx`, `split`);
`read_split(name)` turns a file into positional train, validation and test indices.

| split | test set | train : val |
|---|---|---|
| `random_<seed>`, seeds 42, 67, 69, 404, 1337 | a random 10 % | 80 : 10 |
| `oos_styryl`, seed 42 | every molecule matching the SMARTS `c1ccc(cc1)C=C` | 90 : 10 of the rest |

The styryl split is the out-of-sample split of Leon Schlosser's thesis (Project 1). The
files are regenerated byte-identically by `uv run --extra ml python -m ml.training.splits`.

When it is pointed at the experimental dataset instead, the split matters more
than it looks: an earlier audit found ML classifiers on this data largely memorizing
per-photocatalyst λ fingerprints, and a random split reports that memorization as
accuracy. Leave-one-photocatalyst-out is the honest measure, and it is the only split
that makes a model-versus-theory comparison mean anything.

## Credit and origin

**The model code here is Leon Schlosser's work, not this repository's.** It comes from
[le-schlo/pyrates](https://github.com/le-schlo/pyrates) and is MIT licensed; the licence
travels with it as `LICENSE-pyrates` and must stay with any redistribution. It lives
here as ordinary first-party code — develop against it, fix it, extend it — but the
authorship is his, and anything published from it should say so. The package was
renamed from `pyrates` to `ml` in this repository; `model/checkpoint.py` aliases the
old dotted path so the committed checkpoint still unpickles, byte-identical.

The code was taken at these commits:

| | source | commit | date |
|---|---|---|---|
| everything | branch `fastapi-webapp` | `c7d5fec279a7d05f03bd2a715d66d2e52bab9fd8` | 2026-07-31 |
| `data/training_dataset_full.csv` | branch `main` | `8b1de3c74b05940c0b0c28d64a2482e2154629d1` | 2026-07-31 |

`fastapi-webapp` is the base because it is the branch the deployed application is built
from, and the only place the trained checkpoint exists. The dataset is added back from
`main` because that branch deletes it. Upstream's `data/dataset.csv` is
`data/training_dataset_full.csv` here, next to the checkpoint, which upstream calls
`model.pt` and this repository `model_final.pt`; both files are byte-identical to
upstream.

The modules were rearranged into runtime and training code without changing them; only
import paths differ. Where upstream code lives here:

| upstream (`pyrates/`) | here (`ml/`) |
|---|---|
| `utilities/dataset/graph_construction.py`, `molecule_dataset.py` | `utils/` |
| `utilities/dataset/data_loader.py` | `training/build_dataset.py` |
| `utilities/metrics/digitize.py`, `jaccard.py`, `ranked_score.py` | `training/metrics.py`, concatenated |
| `model/training.py` | `training/loop.py` |

Not carried over: `utilities/plotting/scatter_plot.py` (a parity plot nothing called) and
`main`'s `examples/train.py`, which despite its name only plots one prediction from a
saved checkpoint.

### Known issue: the two upstream branches have diverged

`main` and `fastapi-webapp` forked at `1ae1a13` on 2026-03-20 and were never merged;
nine commits on `main` are absent from the deployed branch. This does not affect the
physics — `energy/` is byte-identical between the two — but "which branch produced the
published numbers" is a question a reviewer can reasonably ask, and it should be
reconciled before publication.
