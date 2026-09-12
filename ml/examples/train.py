from ml.model.inference import get_predictions
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap

label_map_units = {
    "adiabatic_energy": r"$E_\mathrm{T}$ / kcal mol$^{-1}$",
    "lambda_acceptor_singlet": r"$\lambda^{\mathrm{S}\to \mathrm{T}}$ / kcal mol$^{-1}$",
    "lambda_donor_triplet": r"$\lambda^{\mathrm{T}\to \mathrm{S}}$ / kcal mol$^{-1}$",
    "spin_population": "Spin population"
}

custom_colors = [
    "#7E9DDD",  # Medium blue
    "#DEEBF4",  # Pale blue
    "#C0C0C0"   # Silver grey
]

# Set color cycle for plots
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=custom_colors)

fontsize = 26

plt.rcParams.update({
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "gray",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.edgecolor": "black",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": fontsize,
    "xtick.labelsize": fontsize * 0.8,
    "ytick.labelsize": fontsize * 0.8,
    "legend.frameon": False,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.direction": "out",
    "ytick.direction": "out",
})


def _build_cmap():
    pos = 0.25
    return LinearSegmentedColormap.from_list(
        "custom",
        [
            (0.0, "#7E9DDD"),
            (pos * 0.8, "#DEEBF4"),
            (1.0, "#D5C9EF"),
        ],
    )


def plot_mode_distribution(
    mu: np.ndarray,
    sigma: np.ndarray,
    pi: np.ndarray,
    predicted_values: np.ndarray,
    prediction_uncertainty: np.ndarray,
    target_names: list,
    molecule_idx: int = 0,
    main_title: str = None,
    save_path: str = None,
):
    n_targets = len(target_names)
    assert n_targets == 3, "require 3 targets"

    fig, axes = plt.subplots(
        1, 3,
        figsize=(20, 12),
        sharey=True,
    )

    # Top-left reserviert für Molekülstruktur
    axes[0, 0].axis("off")

    # Reihenfolge A, B, C = top-right, bottom-left, bottom-right
    plot_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]

    # sharey blendet ytick-Labels aus, für A wieder einschalten
    plot_axes[0].tick_params(axis="y", labelleft=True)

    if main_title:
        fig.suptitle(main_title, fontsize=fontsize)

    cmap = _build_cmap()

    for target_idx, ax in enumerate(plot_axes):
        ax.text(
            -0.08,
            1.05,
            chr(65 + target_idx),
            transform=ax.transAxes,
            fontsize=fontsize * 0.8,
            fontweight="bold",
            va="top",
            ha="left"
        )

        mu_mol = mu[molecule_idx, target_idx, :]
        sigma_mol = sigma[molecule_idx, target_idx, :]
        pi_mol = pi[molecule_idx, target_idx, :]
        predicted_value = predicted_values[molecule_idx, target_idx]
        uncertainty = prediction_uncertainty[molecule_idx, target_idx]

        n_components = len(mu_mol)

        x_min = float(np.min(mu_mol - 4 * sigma_mol))
        x_max = float(np.max(mu_mol + 4 * sigma_mol))

        x_range = np.linspace(
            x_min,
            x_max,
            1000
        )

        PI_THRESHOLD = 0.01

        sorted_indices = np.argsort(pi_mol)[::-1]

        visible_indices = [
            i for i in sorted_indices
            if pi_mol[i] > PI_THRESHOLD
        ]

        for rank, i in enumerate(visible_indices):
            y = (
                norm.pdf(
                    x_range,
                    mu_mol[i],
                    sigma_mol[i]
                )
                * pi_mol[i]
            )

            color_val = rank / max(n_components - 1, 1)

            ax.fill_between(
                x_range,
                y,
                alpha=0.8,
                color=cmap(color_val),
                label=fr"Mode {rank + 1} ($\pi$={pi_mol[i]:.2f})"
            )

        y_combined = np.sum(
            [
                norm.pdf(
                    x_range,
                    mu_mol[i],
                    sigma_mol[i]
                ) * pi_mol[i]
                for i in range(n_components)
            ],
            axis=0,
        )

        ax.plot(
            x_range,
            y_combined,
            color="black",
            linewidth=2,
            label="Combined PDF"
        )

        def fmt(x):
            return f"{x:.2f}" if abs(x) < 10 else f"{x:.1f}"

        ax.axvline(
            predicted_value,
            color="#6A1B9A",
            linestyle="-",
            linewidth=2.5,
            label=fr"Predicted: {fmt(predicted_value)} kcal mol$^{{-1}}$"
        )

        lower_bound = predicted_value - uncertainty
        upper_bound = predicted_value + uncertainty

        ax.axvline(
            lower_bound,
            color="#C0C0C0",
            linestyle=":",
            linewidth=1.5
        )

        ax.axvline(
            upper_bound,
            color="#C0C0C0",
            linestyle=":",
            linewidth=1.5
        )

        ax.axvspan(
            lower_bound,
            upper_bound,
            color="#C0C0C0",
            alpha=0.2,
            label=fr"$\sigma_{{\mathrm{{pred}}}}$: $\pm${fmt(uncertainty)} kcal mol$^{{-1}}$"
        )

        ax.set_xlabel(
            label_map_units.get(
                target_names[target_idx],
                target_names[target_idx]
            ),
            fontsize=fontsize,
        )

        if target_idx in (0, 1):
            ax.set_ylabel(
                r"Weighted PDF",
                fontsize=fontsize
            )

        leg = ax.legend(
            fontsize=fontsize * 0.6,
            loc="best"
        )

        leg.get_frame().set_facecolor("#DEEBF4")
        leg.get_frame().set_boxstyle("round")
        leg.get_frame().set_alpha(0.2)

    plt.tight_layout(
        rect=(0, 0.03, 1, 0.95)
    )

    if save_path:
        plt.savefig(
            save_path,
            dpi=1200,
            bbox_inches="tight"
        )

        print(f"Plot saved to {save_path}")

    plt.show()


smiles = ["c1ccccc1"]

GRAPH_TARGETS = [
    "adiabatic_energy",
    "lambda_acceptor_singlet",
    "lambda_donor_triplet"
]

NODE_TARGETS = [
    "spin_population"
]

ALL_TARGETS = GRAPH_TARGETS + NODE_TARGETS


predictions, uncertainties, mode_weights, mu, sigma, pi = get_predictions(
    smiles=smiles,
    model_path=Path(
        "/path_to_model/model.pt"
    ),
    device="cuda:0"
)

plot_mode_distribution(
    mu,
    sigma,
    pi,
    predictions,
    uncertainties,
    molecule_idx=0,
    target_names=GRAPH_TARGETS
)