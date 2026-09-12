import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def plot_mode_distribution(
    mu: np.ndarray,
    sigma: np.ndarray,
    pi: np.ndarray,
    predicted_values: np.ndarray,
    prediction_uncertainty: np.ndarray,
    molecule_idx: int,
    target_names: list,
    main_title: str = None,
    target_values: np.ndarray = None,
    save_path: str = None
):
    """
    Visualizes the mixture density network output for a single molecule across multiple target properties using subplots.

    Args:
        mu (np.ndarray): Mean values from the MDN of shape [n_molecules, n_targets, n_components].
        sigma (np.ndarray): Standard deviation values from the MDN of shape [n_molecules, n_targets, n_components].
        pi (np.ndarray): Mixture weights from the MDN of shape [n_molecules, n_targets, n_components].
        predicted_values (np.ndarray): The predicted values (e.g., mixture mean) for this molecule of shape [n_targets].
        prediction_uncertainty (np.ndarray): The prediction uncertainty for this molecule.
        molecule_idx (int): The index of the molecule to plot.
        target_names (list): A list of names for the target properties for the subplot titles.
        main_title (str): The main title for the entire figure.
        target_values (np.ndarray, optional): The actual target values for this molecule of shape [n_targets]. Defaults to None.
        save_path (str, optional): If provided, the plot will be saved to this path. Defaults to None.
    """
    n_targets = len(target_names)
    fig, axes = plt.subplots(1, n_targets, figsize=(10 * n_targets, 6), sharey=True)
    if n_targets == 1:
        axes = [axes]

    if main_title:
        fig.suptitle(main_title, fontsize=16)

    # Create a colormap (defined outside loop for use in colorbar)
    cmap = plt.get_cmap("viridis")

    for target_idx, ax in enumerate(axes):
        # Select the data for the specific molecule and target
        mu_mol = mu[molecule_idx, target_idx, :]
        sigma_mol = sigma[molecule_idx, target_idx, :]
        pi_mol = pi[molecule_idx, target_idx, :]
        target_value = target_values[target_idx] if target_values is not None else None
        predicted_value = predicted_values[molecule_idx, target_idx]
        uncertainty = prediction_uncertainty[molecule_idx,target_idx]

        n_components = len(mu_mol)

        # Create a range for the x-axis
        x_min = min(mu_mol - 4 * sigma_mol)
        x_max = max(mu_mol + 4 * sigma_mol)
        x_range = np.linspace(x_min, x_max, 1000)


        # Sort components by pi value for better visualization
        sorted_indices = np.argsort(pi_mol)

        # Plot each Gaussian component scaled by its pi value
        for i in sorted_indices:
            mu_i = mu_mol[i]
            sigma_i = sigma_mol[i]
            pi_i = pi_mol[i]

            # Generate the PDF for the component
            y = norm.pdf(x_range, mu_i, sigma_i) * pi_i

            # Use pi value to determine color
            color_val = (pi_i - pi_mol.min()) / (pi_mol.max() - pi_mol.min() + 1e-8)
            ax.fill_between(x_range, y, alpha=0.5, color=cmap(color_val), label=f'Mode {i+1} ($\pi$={pi_i:.2f})')

        # Plot the combined PDF
        y_combined = np.sum([norm.pdf(x_range, mu_mol[i], sigma_mol[i]) * pi_mol[i] for i in range(n_components)], axis=0)
        ax.plot(x_range, y_combined, color='black', linestyle='--', linewidth=2, label='Combined PDF')

        # Add a vertical line for the true target value (only if target_values is provided)
        if target_value is not None:
            ax.axvline(target_value, color='red', linestyle='-', linewidth=2, label=f'True Value: {target_value:.2f}')

        # Add a vertical line for the predicted value
        ax.axvline(predicted_value, color='green', linestyle='--', linewidth=2, label=f'Predicted Value: {predicted_value:.2f}')

        # Add uncertainty bounds as dotted lines and shaded area
        lower_bound = predicted_value - uncertainty
        upper_bound = predicted_value + uncertainty
        ax.axvline(lower_bound, color='green', linestyle=':', linewidth=1.5)
        ax.axvline(upper_bound, color='green', linestyle=':', linewidth=1.5)
        y_lim = ax.get_ylim()
        ax.fill_betweenx(y_lim, lower_bound, upper_bound, color='green', alpha=0.15, label=f'Uncertainty: ±{uncertainty:.2f}')
        ax.set_ylim(y_lim)  # Reset y-limits after fill_betweenx

        ax.set_title(f'{target_names[target_idx]}')
        ax.set_xlabel('Value')
        if target_idx == 0:
            ax.set_ylabel('Probability Density (scaled by $\pi$)')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add a shared color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=pi[molecule_idx].min(), vmax=pi[molecule_idx].max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.05, pad=0.02)
    cbar.set_label('$\pi$ (Mode Weight)')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

def plot_mode_distribution_single(
    mu: np.ndarray,
    sigma: np.ndarray,
    pi: np.ndarray,
    predicted_value: float,
    uncertainty: float,
    target_name: str,
    unit: str,
    target_value: float = None,
    theme: str = "light",
):
    if theme == "dark":
        plt.style.use('dark_background')
        bg_color = '#0f172a'
        fg_color = '#f8fafc'
        grid_color = '#334155'
        legend_bg = '#1e293b'
    else:
        plt.style.use('default')
        bg_color = '#ffffff'
        fg_color = '#0f172a'
        grid_color = '#e2e8f0'
        legend_bg = '#ffffff'
        
    fig, ax = plt.subplots(figsize=(8, 5))
    
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    cmap = plt.get_cmap("viridis")

    mu_mol = np.array(mu)
    sigma_mol = np.array(sigma)
    pi_mol = np.array(pi)
    n_components = len(mu_mol)

    x_min = min(mu_mol - 4 * sigma_mol)
    x_max = max(mu_mol + 4 * sigma_mol)
    x_range = np.linspace(x_min, x_max, 1000)

    for i in np.argsort(pi_mol):
        color_val = (pi_mol[i] - pi_mol.min()) / (pi_mol.max() - pi_mol.min() + 1e-8)
        y = norm.pdf(x_range, mu_mol[i], sigma_mol[i]) * pi_mol[i]
        ax.fill_between(x_range, y, alpha=0.5, color=cmap(color_val), label=f'Mode {i+1} (€={pi_mol[i]:.2f})')

    y_combined = np.sum([norm.pdf(x_range, mu_mol[i], sigma_mol[i]) * pi_mol[i] for i in range(n_components)], axis=0)
    ax.plot(x_range, y_combined, color=fg_color, linestyle='--', linewidth=1.5, label='Combined PDF')

    if target_value is not None:
        ax.axvline(target_value, color='#ef4444', linewidth=1.5, label=f'True: {target_value:.2f}')

    ax.axvline(predicted_value, color='#10b981', linestyle='--', linewidth=1.5, label=f'Predicted: {predicted_value:.2f}')

    lower, upper = predicted_value - uncertainty, predicted_value + uncertainty
    ax.axvline(lower, color='#10b981', linestyle=':', linewidth=1)
    ax.axvline(upper, color='#10b981', linestyle=':', linewidth=1)
    y_lim = ax.get_ylim()
    ax.fill_betweenx(y_lim, lower, upper, color='#10b981', alpha=0.15, label=f'{uncertainty:.2f}')
    ax.set_ylim(y_lim)

    ax.set_title(target_name, color=fg_color)
    ax.set_xlabel(f'Value [{unit}]', color=fg_color)
    ax.set_ylabel('Density', color=fg_color)
    ax.tick_params(colors=fg_color)
    for spine in ax.spines.values():
        spine.set_color(grid_color)

    ax.legend(fontsize=7, facecolor=legend_bg, edgecolor=grid_color)
    ax.grid(True, linestyle='--', linewidth=0.5, color=grid_color)
    fig.tight_layout()

    # Reset default style so it doesn't affect other global scripts if any
    plt.style.use('default')
    
    return fig