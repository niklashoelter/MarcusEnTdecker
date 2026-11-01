import torch

def compute_mixture_mean(mu, sigma, pi):
    """
    Compute the mixture expectation (weighted mean) prediction.

    Args:
        mu: Mean values [batch_size, output_dim, n_components]
        sigma: Standard deviations [batch_size, output_dim, n_components]
        pi: Mixture weights [batch_size, output_dim, n_components]

    Returns:
        mean: Mixture expectation [batch_size, output_dim]
        uncertainty: Total standard deviation [batch_size, output_dim]
    """
    # Mixture expectation: sum_k pi_k * mu_k
    mean = torch.sum(pi * mu, dim=-1)

    # Total variance: E[Var] + Var[E]
    # E[Var] = sum_k pi_k * sigma_k^2
    # Var[E] = sum_k pi_k * mu_k^2 - (sum_k pi_k * mu_k)^2
    var_within = torch.sum(pi * sigma ** 2, dim=-1)
    var_between = torch.sum(pi * mu ** 2, dim=-1) - mean ** 2
    total_var = var_within + var_between
    uncertainty = torch.sqrt(total_var)

    return mean, uncertainty

def compute_max_mode(mu, sigma, pi):
    """
    Compute the prediction from the mode with the highest weight (pi).

    Args:
        mu: Mean values [batch_size, output_dim, n_components]
        sigma: Standard deviations [batch_size, output_dim, n_components]
        pi: Mixture weights [batch_size, output_dim, n_components]

    Returns:
        max_mu: Mean of the highest-weight mode [batch_size, output_dim]
        max_sigma: Sigma of the highest-weight mode [batch_size, output_dim]
    """
    max_pi_indices = torch.argmax(pi, dim=-1, keepdim=True)

    # Gather the corresponding mu and sigma
    max_mu = torch.gather(mu, -1, max_pi_indices).squeeze(-1)
    max_sigma = torch.gather(sigma, -1, max_pi_indices).squeeze(-1)

    return max_mu, max_sigma
