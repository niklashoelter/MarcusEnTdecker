# benchmark — physical-organic theory as a classifier

How well do adiabatic thermodynamics, Marcus theory, Rehm–Weller/Balzani, and the DvTE
distribution predict whether a pair quenches? Each is scored against the same ground
truth, `is_quencher = k_q > 1e7 M-1 s-1`, on the same pairs.

Reads `../dataset/pairs.csv`. Writes tables and parity plots into `results/`.

## Status

Not yet ported. The working implementation lives in
`../../phys_org_theories/`, which is scientifically sound but not publishable as it
stands: no version control, a committed virtual environment, `sys.path` manipulation
in place of packaging, and — the blocking problem — importing `constants` copies four
XLSX files in from a sibling directory as a side effect of the import. That makes the
results depend on a path outside the project and on whatever state that path is in.

The port is mechanical: replace the data-access layer with a read of
`dataset/pairs.csv`, keep the theory functions unchanged, and verify the benchmark
numbers come out the same. The physics needs no revision.

## What carries over unchanged

The λ-scaling result, which is the most interesting thing in the benchmark. DFT
harmonic λ is the reorganization energy of *all* modes between the S₀ and T₁
geometries, while the classical Marcus barrier `(ΔG+λ)²/4λ` only applies to modes with
ℏω ≪ k_BT. Three independent effects each predict a downward correction:

1. **Nuclear tunneling.** High-frequency modes (C–H, C–C stretches at ℏω ≫ k_BT)
   tunnel rather than contribute classically, and carry 30–60 % of λ_DFT.
2. **Dexter vs Coulombic coupling.** Triplet–triplet transfer proceeds by double
   electron exchange, which is short-range and insensitive to the outer-sphere solvent
   reorganization that λ_DFT includes (~20–40 % of λ_DFT in acetonitrile).
3. **DFT + PCM over-prediction.** Continuum-solvated DFT over-predicts outer-sphere λ
   by 1.5–2× against experimental electron-transfer benchmarks.

Combined these predict an effective α ≈ 0.3–0.5. The empirical optimum on this dataset
is α ≈ 0.35–0.45 for Marcus and 0.25–0.30 for Rehm–Weller, squarely inside the
predicted range, and worth +15 to +20 MCC over the adiabatic rule. Stability: the
10-seed 80/20 train–test MCC gap is 0.009, and 25-fold leave-one-photocatalyst-out
picks the same α on every fold.

That is a physical correction, not a fitted parameter, and the argument for it should
survive the port intact.

## Two open items the port must not bury

**PC24 is currently dropped by default.** Its DFT λ_T→S came out at 0.89 kcal/mol,
impossible for an Ir(ppy) complex — the singlet and triplet geometries collapsed onto
each other. A near-zero λ flattens the Marcus barrier, and PC24 alone produced roughly
30 false positives. Excluding it costs about 2 MCC across every method equally, so it
does not change any conclusion, but a paper cannot silently drop 1 of 27
photocatalysts. It needs either a re-optimization or an explicit paragraph in the SI.

**PC19 falls out entirely**, with imaginary frequencies in both the S₀ and the reduced
optimization. It has a triplet energy but no reorganization energies and no potentials.

Both depend on photocatalyst DFT job files that are not yet in this repository.
