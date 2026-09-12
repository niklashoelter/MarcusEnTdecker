import torch
from ml.model.ffn.mdn_utils import compute_mixture_mean

def mdn_loss(mu, sigma, pi, target):
    target = target.unsqueeze(-1)

    log_prob = -0.5 * (
        torch.log(2 * torch.pi * sigma**2) +
        ((target - mu)**2) / (sigma**2)
    )

    log_pi = torch.log(pi + 1e-8)
    log_prob_weighted = log_prob + log_pi
    log_sum = torch.logsumexp(log_prob_weighted, dim=-1)

    return -log_sum.mean()

def mdn_loss_per_sample(mu, sigma, pi, target):
    target = target.unsqueeze(-1)

    log_prob = -0.5 * (
        torch.log(2 * torch.pi * sigma**2) +
        ((target - mu)**2) / (sigma**2)
    )

    log_pi = torch.log(pi + 1e-8)
    log_prob_weighted = log_prob + log_pi
    log_sum = torch.logsumexp(log_prob_weighted, dim=-1)

    return -log_sum


def calibration_loss(mu, sigma, pi, target):
    """
    Encourages correlation between predicted uncertainty and actual error.
    Minimizes MSE(|error|, uncertainty), so the model learns to predict
    high uncertainty when the error is large and low uncertainty when accurate.

    Args:
        mu: Mean values [batch_size, output_dim, n_components]
        sigma: Standard deviations [batch_size, output_dim, n_components]
        pi: Mixture weights [batch_size, output_dim, n_components]
        target: Target values [batch_size, output_dim]

    Returns:
        Scalar calibration loss
    """
    mean, uncertainty = compute_mixture_mean(mu, sigma, pi)
    abs_error = torch.abs(target - mean.detach())  # detach mean so gradients only calibrate uncertainty
    return torch.mean((abs_error - uncertainty) ** 2)


def calibration_loss_per_sample(mu, sigma, pi, target):
    """
    Per-sample calibration loss for node-level aggregation.
    Returns per-sample loss [batch_size, output_dim].
    """
    mean, uncertainty = compute_mixture_mean(mu, sigma, pi)
    abs_error = torch.abs(target - mean.detach())
    return (abs_error - uncertainty) ** 2


def sharpness_penalty(mu, sigma, pi, target):
    """
    Asymmetric penalty for overconfident predictions: penalizes cases where
    the predicted uncertainty is smaller than the actual error.
    Underconfident (conservative) predictions are not penalized.

    L = mean(max(0, |error| - uncertainty)^2)

    Args:
        mu: Mean values [batch_size, output_dim, n_components]
        sigma: Standard deviations [batch_size, output_dim, n_components]
        pi: Mixture weights [batch_size, output_dim, n_components]
        target: Target values [batch_size, output_dim]

    Returns:
        Scalar sharpness penalty
    """
    mean, uncertainty = compute_mixture_mean(mu, sigma, pi)
    abs_error = torch.abs(target - mean.detach())
    overconfidence = torch.clamp(abs_error - uncertainty, min=0.0)
    return torch.mean(overconfidence ** 2)


def sharpness_penalty_per_sample(mu, sigma, pi, target):
    """
    Per-sample sharpness penalty for node-level aggregation.
    Returns per-sample loss [batch_size, output_dim].
    """
    mean, uncertainty = compute_mixture_mean(mu, sigma, pi)
    abs_error = torch.abs(target - mean.detach())
    overconfidence = torch.clamp(abs_error - uncertainty, min=0.0)
    return overconfidence ** 2
