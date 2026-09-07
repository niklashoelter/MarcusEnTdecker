"""
Python API for predicting triplet energy transfer rates from structure alone.

    from ml.predict import RatePredictor

    predictor = RatePredictor()
    result = predictor.predict(photocatalyst=PC_SMILES, quencher=Q_SMILES)

    result.dG_kcal          driving force
    result.dG_barrier_kcal  Marcus activation barrier
    result.k_ent            rate constant, M-1 s-1
    result.quenches         k_ent above the detection threshold
    result.uncertainty      from the mixture density network

There is no command-line interface. This is a library: import it.

WHAT IT COMPOSES
    Nothing here re-implements the model or the theory. Both come from the vendored
    upstream package in `upstream/`, and this module is the seam between them:

        upstream prediction.predict(smiles)        graph -> E_T, lambda_S_T, lambda_T_S
        upstream marcus_theory                     properties -> dG, dG barrier
        Balzani rate expression (below)            barrier -> k

    Keeping the barrier in upstream means the web application, the paper's numbers and
    this API all compute it the same way. Only the last step is written here, because
    upstream stops at the barrier and does not combine it with diffusion.

THE RATE EXPRESSION

    k_EnT = k_diff / (1 + exp(dG/RT) + 0.25 * exp(dG_barrier/RT))

    The three terms are diffusion, thermodynamic back-reaction and the activation
    barrier. It reduces to k_diff when transfer is fast and strongly exergonic, which
    is the physically right ceiling -- a bimolecular rate cannot exceed the rate at
    which the two molecules meet. A bare Eyring rate has no such ceiling and will
    happily return 1e14 M-1 s-1, so it is offered as `k_eyring` for comparison but is
    not what `k_ent` reports.

LAMBDA SCALING
    `lambda_scale` multiplies the DFT-like reorganization energies the model predicts
    before they reach the barrier. It defaults to 1.0, the raw prediction.

    On this dataset the benchmark finds an optimum near 0.35-0.45 for Marcus. That is
    not a fudge factor: the classical Marcus barrier applies only to modes with
    energy well below k_BT, while a harmonic reorganization energy sums over all of
    them. High-frequency modes tunnel rather than contribute classically, exchange
    coupling is insensitive to outer-sphere solvent reorganization, and continuum
    solvation over-predicts the outer-sphere part. Each effect pushes the same way,
    and together they predict an effective scale of roughly 0.3-0.5.

    Set it deliberately. The default is left at 1.0 so that nothing is silently
    scaled, but 1.0 is known to over-estimate the barrier.

NOT YET RUN
    This module has not been executed end to end. It needs `uv sync --extra ml`, which
    pulls torch, torch-geometric and rdkit, and it loads a 69 MB checkpoint. The calls
    it makes are taken from the upstream source rather than guessed, but the first run
    should be checked against a few pairs in `../dataset/pairs.csv` before any number
    from it is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# Gas constant in kcal mol-1 K-1.
R_KCAL = 1.987204259e-3

# Acetonitrile diffusion limit, M-1 s-1. A solvent property, not a fitted parameter.
K_DIFF_ACETONITRILE = 2.0e10

# A pair counts as quenching above this rate constant. Defined once in `dataset/`;
# repeated here so the API can answer without loading the dataset.
K_Q_THRESHOLD = 1.0e7

DEFAULT_TEMPERATURE_K = 298.15


@dataclass(frozen=True)
class MolecularProperties:
    """What the model predicts for one structure. Energies in kcal/mol."""

    smiles: str
    E_T: float
    E_T_uncertainty: float
    lambda_S_T: float
    lambda_S_T_uncertainty: float
    lambda_T_S: float
    lambda_T_S_uncertainty: float
    spin_density: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class RatePrediction:
    """A predicted rate constant for one photocatalyst-quencher pair."""

    photocatalyst: str
    quencher: str

    dG_kcal: float
    dG_barrier_kcal: float
    dG_barrier_reverse_kcal: float

    k_ent: float
    k_eyring: float
    quenches: bool

    uncertainty: float
    symmetric_approximation: bool

    photocatalyst_properties: MolecularProperties
    quencher_properties: MolecularProperties

    @property
    def log10_k_ent(self) -> float:
        return math.log10(self.k_ent) if self.k_ent > 0 else float("-inf")


class RatePredictor:
    """Predicts energy transfer rate constants from two SMILES strings.

    The model is loaded once, on the first prediction, and reused. Properties are
    cached per structure, so screening many quenchers against one photocatalyst costs
    one forward pass for the photocatalyst rather than one per pair.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        device: str = "cpu",
        temperature: float = DEFAULT_TEMPERATURE_K,
        k_diff: float = K_DIFF_ACETONITRILE,
        lambda_scale: float = 1.0,
        quenching_threshold: float = K_Q_THRESHOLD,
    ):
        self.model_path = model_path
        self.device = device
        self.temperature = temperature
        self.k_diff = k_diff
        self.lambda_scale = lambda_scale
        self.quenching_threshold = quenching_threshold
        self._cache: dict[str, MolecularProperties] = {}

    # -- properties -------------------------------------------------------------

    def properties(self, smiles: Sequence[str]) -> list[MolecularProperties]:
        """Predict E_T and both reorganization energies for each structure.

        Every structure not already cached is sent to the model in a single batch.
        """
        wanted = list(smiles)
        missing = [s for s in dict.fromkeys(wanted) if s not in self._cache]

        if missing:
            from pyrates.webapp.science_utils.prediction import predict as _predict

            frame = _predict(missing)
            for _, row in frame.iterrows():
                self._cache[row["smiles"]] = MolecularProperties(
                    smiles=row["smiles"],
                    E_T=float(row["triplet_energy_adiabatic"]),
                    E_T_uncertainty=float(row["triplet_energy_adiabatic_uncertainty"]),
                    lambda_S_T=float(row["lamda_S_T"]),
                    lambda_S_T_uncertainty=float(row["lamda_S_T_uncertainty"]),
                    lambda_T_S=float(row["lamda_T_S"]),
                    lambda_T_S_uncertainty=float(row["lamda_T_S_uncertainty"]),
                    spin_density=list(row["spin_map"]),
                )

        return [self._cache[s] for s in wanted]

    # -- rates ------------------------------------------------------------------

    def predict(self, photocatalyst: str, quencher: str) -> RatePrediction:
        """Predict the rate constant for one pair."""
        return self.predict_many([(photocatalyst, quencher)])[0]

    def predict_many(self, pairs: Iterable[tuple[str, str]]) -> list[RatePrediction]:
        """Predict rate constants for many pairs, with one model pass per structure."""
        pairs = list(pairs)
        if not pairs:
            return []

        structures = list(dict.fromkeys([s for pair in pairs for s in pair]))
        self.properties(structures)

        return [self._rate(self._cache[pc], self._cache[q]) for pc, q in pairs]

    def _rate(self, pc: MolecularProperties, q: MolecularProperties) -> RatePrediction:
        from pyrates.energy.eyring import eyring_equation
        from pyrates.webapp.science_utils.marcus_theory import (
            calculate_driving_force_and_barrier,
        )

        a = self.lambda_scale
        dG, barrier, barrier_reverse = calculate_driving_force_and_barrier(
            pc_adiabatic_E_T=pc.E_T,
            pc_lambda_S_T=a * pc.lambda_S_T,
            pc_lambda_T_S=a * pc.lambda_T_S,
            quencher_E_T=q.E_T,
            quencher_lambda_S_T=a * q.lambda_S_T,
            quencher_lambda_T_S=a * q.lambda_T_S,
        )

        k_ent = self._balzani(dG, barrier)
        uncertainty = math.hypot(pc.E_T_uncertainty, q.E_T_uncertainty)

        return RatePrediction(
            photocatalyst=pc.smiles,
            quencher=q.smiles,
            dG_kcal=float(dG),
            dG_barrier_kcal=float(barrier),
            dG_barrier_reverse_kcal=float(barrier_reverse),
            k_ent=k_ent,
            k_eyring=float(eyring_equation(barrier, T=self.temperature)),
            quenches=k_ent > self.quenching_threshold,
            uncertainty=uncertainty,
            # calculate_driving_force_and_barrier falls back to the symmetric form
            # when the parabolas do not intersect. It reports that by printing, not
            # by returning, so it is re-derived here from the same condition.
            symmetric_approximation=self._is_symmetric(pc, q, a),
            photocatalyst_properties=pc,
            quencher_properties=q,
        )

    def _balzani(self, dG: float, barrier: float) -> float:
        """k_diff / (1 + exp(dG/RT) + 0.25 exp(barrier/RT)), guarded against overflow."""
        rt = R_KCAL * self.temperature

        def bounded_exp(x):
            # Both terms only ever increase the denominator, so a saturated exponent
            # drives the rate to zero, which is the correct limit.
            return math.exp(min(x / rt, 700.0))

        denominator = 1.0 + bounded_exp(dG) + 0.25 * bounded_exp(barrier)
        return self.k_diff / denominator

    @staticmethod
    def _is_symmetric(pc: MolecularProperties, q: MolecularProperties, a: float) -> bool:
        """Whether upstream would have fallen back to the symmetric approximation."""
        forward = a * (pc.lambda_T_S + q.lambda_S_T)
        reverse = a * (pc.lambda_S_T + q.lambda_T_S)
        dG = q.E_T - pc.E_T
        root = forward * reverse + (forward - reverse) * dG
        return root < 0 or abs(forward - reverse) < 1e-10
