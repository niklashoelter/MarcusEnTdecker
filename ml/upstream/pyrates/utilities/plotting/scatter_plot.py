import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


def plot_predictions(preds, targets, uncertainties=None, mode_weights=None,
                     target_names=None, figsize=None, save_path=None):
    """
    Create scatter plots comparing predictions vs targets for each output dimension.

    Parameters:
    -----------
    preds : np.ndarray
        Predictions, shape [n_samples, n_targets]
    targets : np.ndarray
        Ground truth values, shape [n_samples, n_targets]
    uncertainties : np.ndarray, optional
        Prediction uncertainties, shape [n_samples, n_targets]
    mode_weights : np.ndarray, optional
        Mode weights (pi values) for each prediction, shape [n_samples]
    target_names : list of str, optional
        Names for each target dimension
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated
    save_path : str, optional
        Path to save the figure
    """
    n_targets = preds.shape[1]

    # Default target names if not provided
    if target_names is None:
        target_names = [f'Target {i + 1}' for i in range(n_targets)]

    # Calculate grid dimensions
    n_cols = min(3, n_targets)  # Max 3 columns
    n_rows = int(np.ceil(n_targets / n_cols))

    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i in range(n_targets):
        ax = axes[i]

        # Get data for this target
        pred_i = preds[:, i]
        target_i = targets[:, i]

        # Plot with or without uncertainty
        if uncertainties is not None:
            unc_i = uncertainties[:, i]

            # Color by uncertainty or mode weight
            if mode_weights is not None:
                scatter = ax.scatter(target_i, pred_i, c=mode_weights,
                                     alpha=0.6, s=30, cmap='viridis',
                                     edgecolors='k', linewidth=0.5)
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Mode Weight (π)', rotation=270, labelpad=15)
            else:
                scatter = ax.scatter(target_i, pred_i, c=unc_i,
                                     alpha=0.6, s=30, cmap='plasma',
                                     edgecolors='k', linewidth=0.5)
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Uncertainty (σ)', rotation=270, labelpad=15)

            # Optional: Add error bars
            # ax.errorbar(target_i, pred_i, yerr=unc_i, fmt='none',
            #             alpha=0.2, color='gray', linewidth=0.5)
        else:
            ax.scatter(target_i, pred_i, alpha=0.6, s=30,
                       edgecolors='k', linewidth=0.5)

        # Perfect prediction line
        min_val = min(target_i.min(), pred_i.min())
        max_val = max(target_i.max(), pred_i.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)

        # Calculate metrics
        r2 = r2_score(target_i, pred_i)
        mae = mean_absolute_error(target_i, pred_i)
        rmse = root_mean_squared_error(target_i, pred_i)

        # Add metrics to plot
        metrics_text = f'R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Labels and title
        ax.set_xlabel('True Values', fontsize=11)
        ax.set_ylabel('Predicted Values', fontsize=11)
        ax.set_title(target_names[i], fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # Equal aspect ratio for better visualization
        ax.set_aspect('equal', adjustable='box')

    # Hide unused subplots
    for i in range(n_targets, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig, axes