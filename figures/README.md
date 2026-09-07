# figures

Every figure in the paper and the SI, regenerated from `../dataset/` and
`../benchmark/results/`. One script per figure, named for the figure it makes.

The rule is that no figure is drawn from a number typed by hand. If a figure cannot be
produced from a processed table, the table is missing, not the figure.

## Status

Not started.

## Known issue carried over

The current Figure 3 claims a Sandros / Rehm–Weller reading that the data does not
support: roughly half the pairs it draws on are left-censored, and a fit through
censored points treated as measurements is not a fit. Whatever replaces it has to
either restrict to the uncensored subset and say so, or show the censoring explicitly
(arrows at the floor, which is photocatalyst-specific). This is a scientific decision
about what the figure claims, not a plotting problem.
