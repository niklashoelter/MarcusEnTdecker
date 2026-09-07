# Triplet energy transfer kinetics — data, benchmarks and models

Companion repository for *A Hybrid Machine-Learning and Marcus Theory Approach for
Predicting Triplet Energy Transfer Kinetics*.

It holds everything needed to regenerate the paper's numbers from instrument output:
the experimental and computational data with the scripts that process them, the
assembled pair-level dataset, the physical-organic theory benchmarks, and the training
and evaluation of the machine-learning models.

## The screen

27 photocatalysts × 48 quenchers = **1296 pairs**, complete. Automated Stern–Volmer
quenching in acetonitrile on an Opentrons Flex and a BMG VantaStar plate reader;
96-well glass-coated plates, six-point dilution series.

## Layout

```
data/           instrument output and DFT job files, and the scripts that process them
dataset/        the pair-level table -- where the processed data becomes one row per pair
benchmark/      physical-organic theory as classifiers: adiabatic, Marcus, Rehm-Weller, DvTE
ml/             training and evaluation of the models, on top of pyrates
figures/        every figure in the paper, regenerated from processed data
paper/          tables destined for the manuscript and SI, generated rather than typed
deploy/         the EnTdecker web application deployment
```

Dependencies run strictly one way:

```
data  ->  dataset  ->  { benchmark, ml }  ->  figures, paper
```

Nothing reaches sideways or backwards. A script reads only from directories to its
left, so any number in the paper can be traced to the instrument file it came from by
following that chain in one direction.

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
archived to Zenodo rather than versioned here, and everything downstream of it is
already committed as processed CSVs — so the repository is fully usable without it.
You only need it to re-derive the computational tables from scratch. See
`data/computational_data_raw/README.md`.

## State

| stage | state |
|---|---|
| `data/` experimental | **complete and verified** — 5 of 6 scripts written, all round-trip byte-identical |
| `data/` computational | **quenchers complete and verified**; photocatalyst job files not yet available |
| `dataset/` | in progress |
| `benchmark/` | to be ported from `phys_org_theories` |
| `ml/` | to be written against a pinned pyrates release |
| `figures/`, `paper/` | not started |

Each directory's own README says what is in it, what produced it, and what is known to
be wrong with it.

## Relationship to pyrates

The machine-learning models are not implemented here. They live in
[pyrates](https://github.com/le-schlo/pyrates) (Leon Schlosser, MIT), which is also
what the EnTdecker web application runs. This repository depends on a pinned pyrates
release and adds only the training configuration, the evaluation against this dataset,
and the deployment manifests.

Keeping the two apart is deliberate: pyrates is a library that will keep developing,
while this repository is frozen at publication so that the paper's numbers stay
reproducible.

## Citing

See `CITATION.cff`. Code is MIT (`LICENSE`); data is CC BY 4.0 (`LICENSE-DATA`).
