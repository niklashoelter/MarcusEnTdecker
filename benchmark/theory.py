# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""
What every theory module in this folder has to look like, and the physics they share.

Reads   nothing
Writes  nothing

THE INTERFACE
    A theory module is a plain Python module exposing the following. The scoring harness
    knows nothing else about it, so adding a theory means adding a file and one import.

        NAME               str, the short name used in the output tables
        DESCRIPTION        one line, printed in the run header
        required_columns() the pair-table columns it cannot run without
        prepare(df, **kw)  heavy per-pair computation, done once for the whole grid.
                           Takes k_diff, mean_type and legacy_pairing; a module with no
                           use for them swallows them, so the harness need not know
                           which is which.
        grid()             the Configs to score as classifiers -- every parameter
                           setting, not just the one that will win
        classify(prep, cfg)       -> bool array, one prediction per row
        rate_grid()               -> the Configs that state a rate constant, which may
                                     be an empty list. It is a separate grid because a
                                     rate does not depend on the classification
                                     threshold applied to it, and scoring the same rate
                                     once per threshold would fill the table with copies.
        predicted_rate(prep, cfg) -> float array of k_q in M-1 s-1
        note(prep)                -> optional one line printed after the description,
                                     for anything fitted at prepare time

    `prepare` exists so a 16-value lambda sweep crossed with 30 barrier thresholds costs
    16 barrier evaluations, not 480. `classify` only thresholds what `prepare` computed.

FREE PARAMETERS ARE DECLARED, NOT HIDDEN
    Every Config says how many free parameters it carries. The adiabatic rule carries
    zero; a Marcus barrier gate at a swept threshold and a swept lambda scale carries
    two. This matters because the harness picks the best threshold by MCC on the same
    data it then reports the MCC of, which flatters a tunable method against a rule that
    cannot be tuned. `n_free` is what lets the output separate the two, and it is why
    the harness also reports a leave-one-photocatalyst-out score for anything with
    `n_free > 0`.

UNITS
    Energies kcal/mol throughout, rate constants M-1 s-1, potentials V vs SCE. The
    column headers in the pair table carry their units after " / " and this module keeps
    to them.
"""

from dataclasses import dataclass, field

import numpy as np

# Gas constant in kcal/(mol K), and the temperature every measurement was made at.
R_KCAL = 1.9872e-3
TEMPERATURE = 298.15
KBT = R_KCAL * TEMPERATURE

# The Smoluchowski bimolecular diffusion limit for acetonitrile. A solvent property, not
# a tunable of the theory, so it is not swept.
#
# The original implementation carried a second "calibrated" scenario at 5e8, an effective
# pre-factor fitted to the data. It is not ported. Its own audits withdrew it: the
# k_diff sweep from 1e8 to 5e10 landed on the "do not retest" list because "barrier
# methods don't use it; rate methods pick a compensating log_k cutoff either way" -- the
# apparent gain was absorbed by the rate methods' own tuned threshold. Only the
# literature value is used here, and --k-diff exists for sensitivity checks.
K_DIFF_ACETONITRILE = 2.0e10


@dataclass(frozen=True)
class Config:
    """One scoreable setting of one theory."""

    family: str          # methods that differ only in a swept parameter share a family
    name: str            # unique within the run
    n_free: int          # free parameters selected by the harness on the scored data
    params: dict = field(default_factory=dict, compare=False)


def balzani_rate(dG, dG_activation, k_diff=K_DIFF_ACETONITRILE, temperature=TEMPERATURE):
    """k_EnT from a driving force and a barrier, via the Balzani rate equation.

        k_EnT = k_diff / (1 + exp(dG/kT) + 0.25 * exp(dG_act/kT))

    Balzani et al., JACS 1980, 102, 2152; the form used in Scriven, Troian-Gautier et
    al., JACS 2026, 148, 8993, Eq. 3-4. The 0.25 is k_-diff/Z, the Rehm-Weller constant
    for acetonitrile, and it multiplies only the activation term -- not the thermodynamic
    equilibrium term, which is where the electron-transfer form of the same equation
    differs.

    Shared by marcus.py and rehm_weller.py, which differ only in how dG_activation is
    obtained. The original implementation had this expression written out identically in
    both files; it is written once here so the two cannot drift apart.

    Exponents are clamped at 500 before exponentiation, as in the original, so a
    strongly endergonic pair returns a very small rate instead of an overflow warning.
    """
    dG = np.asarray(dG, dtype=float)
    dG_activation = np.asarray(dG_activation, dtype=float)
    kbT = R_KCAL * temperature
    exp_dG = np.exp(np.minimum(dG / kbT, 500.0))
    exp_act = np.exp(np.minimum(dG_activation / kbT, 500.0))
    return k_diff / (1.0 + exp_dG + 0.25 * exp_act)


def symmetric_lambda_rxn(lambda_forward, lambda_reverse, mean_type="geometric"):
    """One effective reorganization energy out of the forward and reverse pair.

    'geometric' is sqrt(lambda_f * lambda_r), 'arithmetic' is (lambda_f + lambda_r)/2.
    Geometric is the default because it fit this dataset better; the difference between
    the two is under 0.01 MCC on every subset the original tested, so the choice is not
    load-bearing and the alternative is kept only so that can be re-checked.
    """
    lf = np.asarray(lambda_forward, dtype=float)
    lr = np.asarray(lambda_reverse, dtype=float)
    if mean_type == "geometric":
        return np.sqrt(np.maximum(lf * lr, 0.0))
    if mean_type == "arithmetic":
        return 0.5 * (lf + lr)
    raise ValueError(f"Unknown mean_type {mean_type!r}; expected 'geometric' or 'arithmetic'")


# The reorganization-energy scaling sweep, ported with its justification intact.
#
# alpha = 1.0 is the textbook classical-Marcus limit: the DFT harmonic lambda used as is.
# Smaller alpha applies an effective-lambda correction. Three independent mechanisms each
# predict a 15-40 % reduction from the DFT harmonic lambda to the lambda that belongs in
# the classical barrier (dG + lambda)^2 / 4 lambda, which is only correct for modes with
# hbar w << kB T:
#
#   1. Marcus-Jortner nuclear tunneling. High-frequency C-C and C-H stretches
#      (hbar w > 2000 cm-1, against kB T = 200 cm-1) tunnel through the barrier rather
#      than contributing to it classically, and carry 30-60 % of lambda_DFT for typical
#      organic triplets. Jortner, J. Chem. Phys. 1976, 64, 4860.
#   2. Dexter exchange is short-range. Triplet-triplet transfer proceeds by double
#      electron exchange, which does not couple to the outer-sphere Coulombic solvent
#      reorganization that lambda_DFT includes -- 20-40 % of lambda_DFT in acetonitrile.
#   3. DFT + PCM over-predicts outer-sphere lambda by 1.5-2x against experimental
#      electron-transfer benchmarks. Matyushov, Mol. Phys. 2009, 107, 1027; Blumberger,
#      Chem. Rev. 2015, 115, 11191.
#
# Combined they put the expected alpha near 0.3-0.5, and the empirical optimum on the
# original benchmark landed at 0.35-0.45 for Marcus and 0.25-0.30 for Rehm-Weller.
#
# Two honesty notes that must travel with this number:
#   - The band above is a plausibility argument, not a derivation. It was assembled after
#     the empirical optimum was found, and the original's own two write-ups disagree on
#     its arithmetic.
#   - alpha is selected by maximizing MCC on the data the MCC is then reported for. The
#     original's stability evidence (10-seed 80/20 gap of 0.009 MCC, and 25 of 25
#     leave-one-photocatalyst-out folds picking the same alpha) was collected under a
#     different label threshold and never re-run under the current one. That is why the
#     harness reports the alpha = 1.0 row and the tuned row separately, and a
#     leave-one-photocatalyst-out score alongside the tuned one.
#
# Below 0.25 or above 1.0 is not physically defensible, so the grid stops there.
LAMBDA_SCALES = tuple(round(float(x), 2) for x in np.arange(0.25, 1.001, 0.05))

# Barrier gate thresholds, kcal/mol, and log10 rate thresholds, log10(M-1 s-1).
BARRIER_THRESHOLDS = tuple(round(float(x), 2) for x in np.arange(0.5, 15.01, 0.5))
LOG_RATE_THRESHOLDS = tuple(round(float(x), 2) for x in np.arange(4.0, 10.01, 0.25))


def scale_tag(alpha: float) -> str:
    """The suffix that marks a lambda-scaled variant in a method name."""
    return "" if alpha == 1.0 else f"_a{alpha:g}"
