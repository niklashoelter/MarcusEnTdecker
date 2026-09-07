# ml — training and evaluation

The models themselves are not implemented here. They live in
[pyrates](https://github.com/le-schlo/pyrates) (Leon Schlosser, MIT): graph encoders
(D-MPNN, GAT, GIN, Graph Transformer, GPS), a mixture density network for uncertainty,
multi-task heads, and the loss functions.

This directory holds only what is specific to this paper — the training configuration,
the split, and the evaluation against `../dataset/pairs.csv` — so that the published
numbers can be regenerated without freezing a copy of somebody else's library inside
this repository.

```
configs/     one file per reported model
train.py     fit against dataset/pairs.csv, write to models/
evaluate.py  score against the same held-out split the benchmark uses
models/      trained weights (gitignored — archived with the data)
```

## Pinning

`pyproject.toml` pins pyrates under the `ml` extra:

```
uv sync --extra ml
```

The pin currently points at `main` and **must be changed to a release tag before
submission**. The published numbers depend on the exact version; pyrates is an active
library and will keep moving after this paper is frozen.

## Comparability

The evaluation has to be scored on the same pairs, the same ground truth, and the same
split as `../benchmark/`, or the model-versus-theory comparison in the paper means
nothing. That means:

- ground truth `is_quencher = k_q > 1e7 M-1 s-1`, defined once in `dataset/`
- the same left-censored non-quenchers included as negatives, not dropped
- splitting by photocatalyst, not at random

The last one matters more than it looks. An earlier audit of ML classifiers on this
data found they were largely memorizing per-photocatalyst λ fingerprints; a random
split lets the same photocatalyst appear on both sides and reports that memorization
as accuracy. Leave-one-photocatalyst-out is the honest measure and is what the theory
benchmark already uses.

## Status

Not written. Blocked only on the pinned release; the dataset side is the real
prerequisite.
