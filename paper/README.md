# paper — generated tables

Tables destined for the manuscript and the SI, generated from `../data/` and
`../benchmark/results/` rather than typed.

The manuscript itself is not kept here. This directory produces the tables it needs, so
that a number appearing in the paper and the same number in the dataset cannot drift
apart.

## Status

Not started.

## Known errata to carry into the SI

Three values in the published quencher DFT table contradict the free energies printed
beside them in their own rows. Recomputing from the ORCA outputs gives the value each
ΔG implies:

|     | ΔG published | E published | corrected |
|-----|-------------:|------------:|----------:|
| Q9  |        134.8 |        1.41 |  **1.40** |
| Q14 |        159.0 |        2.46 |  **2.45** |
| Q18 |         44.0 |       −2.54 | **−2.53** |

Four photocatalyst triplet energies in the delivered DFT table
(`properties_photocatalysts_calc.xlsx`) match no job in the delivery: PC8, PC11, PC23 and
PC26 (old 10, 13, 25, 28) differ from their jobs by +2.23, −0.70, +0.61 and −0.55 kcal/mol. `../data/` uses the
job values. See `../data/computational_data_scripts/photocatalyst_dft.py`.

PC numbers here are the new ids, PC1–PC27 (`../data/compound_identification.xlsx`). Leon's Table S2
in the SI uses a third numbering (26 entries, no old PC19, old PC27 in slot 17) and has to
be brought in line with it.

Not an erratum, but the SI methods need it: PC4's τ₀ is fitted from 1500 ns, past a fast
early component, and reproduces the published 1494.54 ns only that way (from the
automatic start it gives 1228.49 ns). See `../data/experimental_data_scripts/lifetimes_tcspc.py`.
