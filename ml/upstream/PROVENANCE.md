# Provenance of this directory

Everything under `ml/upstream/` is a vendored copy of
[le-schlo/pyrates](https://github.com/le-schlo/pyrates) by Leon Schlosser, MIT
licensed (see `LICENSE`). It is here so that the models the paper reports, and the
web application that serves them, stay reproducible after the upstream library moves
on.

Do not develop against this copy. Fixes belong upstream and come back here as a
re-vendor.

## What was taken

| | source | commit | date |
|---|---|---|---|
| everything | branch `fastapi-webapp` | `c7d5fec279a7d05f03bd2a715d66d2e52bab9fd8` | 2026-07-31 |
| `data/dataset.csv` | branch `main` | `8b1de3c74b05940c0b0c28d64a2482e2154629d1` | 2026-07-31 |
| `examples/train.py` | branch `main` | `8b1de3c74b05940c0b0c28d64a2482e2154629d1` | 2026-07-31 |

`fastapi-webapp` is the base because it is the branch the deployed application is
built from, and it is the only place the trained checkpoint exists:
`pyrates/webapp/data/model.pt`, 69 MB. The two files above are added back because
that branch deletes them and they are what training runs from.

## The one modification

`.gitignore` was deleted. It listed `data/`, which in this repository would have
silently excluded both `data/dataset.csv` and the 69 MB checkpoint at
`pyrates/webapp/data/model.pt` — the two files most necessary to reproduce anything.
Upstream committed them before that rule existed, so the contradiction is invisible
there and only bites on a fresh copy.

Nothing else was changed. No import was rewritten, no path was adjusted, no code was
reformatted.

## The branches have diverged and should be reconciled

`main` and `fastapi-webapp` forked at `1ae1a13` on 2026-03-20 and were never merged.
Nine commits on `main` are absent from the deployed branch, including
`665fd51 update marcus and included dataset`.

The important part is that this does **not** affect the physics: `pyrates/energy/` is
byte-identical between the two branches, so Marcus and Eyring are the same code either
way. The differences are `pyrates/model/inference.py` (+97 lines on the webapp
branch), two plotting modules, `pyproject.toml`, `LICENSE` and `README.md`.

Still worth resolving before submission, because "which branch produced the published
numbers" is a question a reviewer can reasonably ask and the honest answer today is
"the deployed one, which is missing nine commits from main".

## Re-vendoring

```
git clone https://github.com/le-schlo/pyrates.git
cd pyrates && git checkout <tag>
# copy the tree in, drop .gitignore, add back data/dataset.csv and examples/train.py
# if the tag does not carry them, then update the table above
```

Before publication this should point at a release tag rather than a branch head, so
that the version is nameable in the paper.
