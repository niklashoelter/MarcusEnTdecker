"""
Python API for predicting triplet energy transfer rates from structure alone.

    from ml.predict import RatePredictor

    predictor = RatePredictor()
    result = predictor.predict(photocatalyst=photocatalyst_properties(1),
                               quencher=Q_SMILES)

    result.dG_kcal          driving force
    result.dG_barrier_kcal  Marcus activation barrier
    result.k_ent            rate constant, M-1 s-1
    result.quenches         k_ent above the detection threshold
    result.uncertainty      from the mixture density network

There is no command-line interface. This is a library: import it.

WHAT IT COMPOSES
    Nothing here re-implements the model or the theory. Both come from the `pyrates`
    package alongside this file, and this module is the seam between them:

        pyrates prediction.predict(smiles)         graph -> E_T, lambda_S_T, lambda_T_S
        pyrates marcus_theory                      properties -> dG, dG barrier
        Balzani rate expression (below)            barrier -> k

    Keeping the barrier in `pyrates` means the web application, the paper's numbers
    and this API all compute it the same way. Only the last step is written here,
    because `pyrates` stops at the barrier and does not combine it with diffusion.

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

THE MODEL CANNOT SEE THE PHOTOCATALYSTS
    Its element vocabulary is H, B, C, N, O, F, Si, P, S, Cl, Se, Br, Sn, I. Iridium
    (77) and ruthenium (44) are not in it, and encoding one raises `UnknownAtomError`
    rather than returning a poor answer -- which is the right behaviour, but it means
    28 of the 29 screened photocatalysts cannot be passed to it as a structure at all.
    Only PC7, an organic thioxanthone, can.

    This is a property of the training set, not a defect: the model was trained on
    organic triplet sensitizers. So a pair is predicted from two different sources,
    and the API takes them as two different kinds of argument:

        the quencher       a SMILES -> the model         (organic, in the vocabulary)
        the photocatalyst  its DFT properties            (a metal complex)

    `photocatalyst_properties(pc_id)` reads those from
    `../data/computational_data_processed/photocatalyst_dft.csv`. Passing a metal
    complex as a SMILES will fail, loudly, and that is deliberate -- silently falling
    back to something else would hide which half of each prediction is a model output.

    Both arguments accept either form, so an organic photocatalyst can still be given
    as a SMILES and a quencher can be given as measured properties.

MIXING A MODEL PREDICTION WITH A CALCULATION
    A rate from this API is therefore a hybrid: a DFT triplet energy for one partner
    and a predicted one for the other. The two carry different errors. `uncertainty`
    reports only what the model estimates, because a DFT number arrives without one,
    and `uncertainty_covers` says which partners it accounts for. Do not read it as a
    total.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, Union

PHOTOCATALYST_DFT = (Path(__file__).resolve().parent.parent / "data"
                     / "computational_data_processed" / "photocatalyst_dft.csv")

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
    lambda_S_T: float
    lambda_T_S: float
    # None where the properties came from a calculation rather than the model. A DFT
    # number arrives without an uncertainty, and inventing a 0.0 for it would make a
    # hybrid prediction look better determined than it is.
    E_T_uncertainty: float | None = None
    lambda_S_T_uncertainty: float | None = None
    lambda_T_S_uncertainty: float | None = None
    spin_density: list[float] = field(default_factory=list)
    source: str = "model"


#: Either a structure the model can encode, or properties already in hand.
Partner = Union[str, "MolecularProperties"]


def photocatalyst_properties(pc_id: int, path: Path | None = None) -> MolecularProperties:
    """DFT properties for one photocatalyst, in the form `predict` takes.

    The model cannot encode iridium or ruthenium, so this is how a metal-complex
    photocatalyst enters a prediction: from `photocatalyst_dft.csv`, which is derived
    from the deposited ORCA jobs.

    The two reorganization energies map across by which surface they relax on --
    `lambda_S0_to_T1` is the relaxation on the triplet surface from the singlet
    geometry, which is what the Marcus expression calls `lambda_S_T`.
    """
    source = path or PHOTOCATALYST_DFT
    if not source.exists():
        raise FileNotFoundError(
            f"{source} is missing; run "
            f"`uv run data/computational_data_scripts/photocatalyst_dft.py` first"
        )

    with source.open(encoding="utf-8", newline="") as fh:
        row = next((r for r in csv.DictReader(fh) if int(r["pc_id"]) == pc_id), None)

    if row is None:
        raise KeyError(f"no photocatalyst {pc_id} in {source.name}")

    needed = ("E_T / kcal mol-1", "lambda_S0_to_T1 / kcal mol-1",
              "lambda_T1_to_S0 / kcal mol-1")
    absent = [c for c in needed if not row[c]]
    if absent:
        raise ValueError(
            f"PC{pc_id} has no {', '.join(absent)} -- its jobs are incomplete "
            f"({row['note']}). It cannot be used for a rate prediction."
        )

    return MolecularProperties(
        smiles=f"PC{pc_id}",
        E_T=float(row["E_T / kcal mol-1"]),
        lambda_S_T=float(row["lambda_S0_to_T1 / kcal mol-1"]),
        lambda_T_S=float(row["lambda_T1_to_S0 / kcal mol-1"]),
        source="dft",
    )


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

    uncertainty: float | None
    uncertainty_covers: str
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
                    source="model",
                )

        return [self._cache[s] for s in wanted]

    # -- rates ------------------------------------------------------------------

    def predict(self, photocatalyst: Partner, quencher: Partner) -> RatePrediction:
        """Predict the rate constant for one pair.

        Each partner is either a SMILES, which the model is asked about, or a
        `MolecularProperties`, which is used as given. A metal-complex photocatalyst
        has to be the second kind -- see `photocatalyst_properties`.
        """
        return self.predict_many([(photocatalyst, quencher)])[0]

    def predict_many(self, pairs: Iterable[tuple[Partner, Partner]]) -> list[RatePrediction]:
        """Predict rate constants for many pairs, with one model pass per structure.

        Only the partners given as SMILES reach the model, and each distinct one
        reaches it once. Screening 48 quenchers against one photocatalyst whose
        properties are already known therefore costs 48 forward passes, not 49.
        """
        pairs = list(pairs)
        if not pairs:
            return []

        smiles = [p for pair in pairs for p in pair if isinstance(p, str)]
        self.properties(list(dict.fromkeys(smiles)))

        def resolve(partner: Partner) -> MolecularProperties:
            return self._cache[partner] if isinstance(partner, str) else partner

        return [self._rate(resolve(pc), resolve(q)) for pc, q in pairs]

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

        # Only the partners the model spoke about contribute. Combining an estimated
        # uncertainty with an absent one by treating the absent one as zero would
        # understate the spread on every hybrid prediction.
        estimated = [(name, p.E_T_uncertainty) for name, p in
                     (("photocatalyst", pc), ("quencher", q))
                     if p.E_T_uncertainty is not None]
        uncertainty = (math.sqrt(sum(u * u for _, u in estimated)) if estimated else None)
        covers = " and ".join(name for name, _ in estimated) or "neither partner"

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
            uncertainty_covers=covers,
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
        """Whether pyrates would have fallen back to the symmetric approximation."""
        forward = a * (pc.lambda_T_S + q.lambda_S_T)
        reverse = a * (pc.lambda_S_T + q.lambda_T_S)
        dG = q.E_T - pc.E_T
        root = forward * reverse + (forward - reverse) * dG
        return root < 0 or abs(forward - reverse) < 1e-10
