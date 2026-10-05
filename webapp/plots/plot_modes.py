import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

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
        ax.fill_between(x_range, y, alpha=0.5, color=cmap(color_val), label=f'Mode {i+1} (π = {pi_mol[i]:.2f})')

    y_combined = np.sum([norm.pdf(x_range, mu_mol[i], sigma_mol[i]) * pi_mol[i] for i in range(n_components)], axis=0)
    ax.plot(x_range, y_combined, color=fg_color, linestyle='--', linewidth=1.5, label='Combined PDF')

    if target_value is not None:
        ax.axvline(target_value, color='#ef4444', linewidth=1.5, label=f'True: {target_value:.2f}')

    ax.axvline(predicted_value, color='#10b981', linestyle='--', linewidth=1.5, label=f'Predicted: {predicted_value:.2f}')

    lower, upper = predicted_value - uncertainty, predicted_value + uncertainty
    ax.axvline(lower, color='#10b981', linestyle=':', linewidth=1)
    ax.axvline(upper, color='#10b981', linestyle=':', linewidth=1)
    y_lim = ax.get_ylim()
    ax.fill_betweenx(y_lim, lower, upper, color='#10b981', alpha=0.15, label=f'±{uncertainty:.2f}')
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