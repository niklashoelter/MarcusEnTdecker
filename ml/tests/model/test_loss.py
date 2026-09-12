import torch
import pytest
from ml.model.loss import mdn_loss, calibration_loss

def test_mdn_loss_perfect_fit():
    # If the model predicts perfectly, the loss should be minimized.
    # Case: Component 1 has mean=target, sigma small. Component 2 is far away.
    # Pi leans towards component 1.

    target = torch.tensor([[5.0]])
    mu = torch.tensor([[[5.0, 10.0]]])
    sigma = torch.tensor([[[0.1, 1.0]]])
    pi = torch.tensor([[[0.9, 0.1]]])

    loss = mdn_loss(mu, sigma, pi, target)

    # Check that loss is a scalar tensor
    assert loss.ndim == 0

    # Calculate expected loss manually
    # LogProb1 = -0.5 * (log(2*pi*0.1^2) + 0) = -0.5 * log(0.0628) approx -0.5 * -2.76 = 1.38
    # LogProb2 = -0.5 * (log(2*pi*1^2) + (5-10)^2/1) = -0.5 * (1.83 + 25) = -13.4
    # Weighted: log(0.9) + 1.38 vs log(0.1) - 13.4
    # -0.1 + 1.38 = 1.28 vs -2.3 - 13.4 = -15.7
    # LogSumExp(1.28, -15.7) approx 1.28
    # Negative -> -1.28

    # Just checking it runs and gives a reasonable value
    assert loss < 0 # NLL can be negative for continuous distributions (densities > 1)

def test_calibration_loss():
    # Target = 5.0
    # Mean prediction = 5.0 (perfect accuracy)
    # Uncertainty = 1.0
    # Error = 0.0
    # Calibration loss = MSE(|0 - 1|) = 1.0

    target = torch.tensor([[5.0]])
    mu = torch.tensor([[[5.0]]])
    sigma = torch.tensor([[[1.0]]]) # Uncertainty will be 1.0
    pi = torch.tensor([[[1.0]]])

    loss = calibration_loss(mu, sigma, pi, target)

    # mean, uncertainty = compute_mixture_mean(...) => 5.0, 1.0
    # abs_error = |5.0 - 5.0| = 0.0
    # loss = (0.0 - 1.0)^2 = 1.0

    assert torch.isclose(loss, torch.tensor(1.0))

def test_calibration_loss_perfect():
    # Target = 5.0
    # Mean prediction = 6.0 (Error = 1.0)
    # Uncertainty = 1.0
    # Calibration loss = (1.0 - 1.0)^2 = 0.0

    target = torch.tensor([[5.0]])
    mu = torch.tensor([[[6.0]]])
    sigma = torch.tensor([[[1.0]]]) # Uncertainty will be 1.0
    pi = torch.tensor([[[1.0]]])

    loss = calibration_loss(mu, sigma, pi, target)

    assert torch.isclose(loss, torch.tensor(0.0))

