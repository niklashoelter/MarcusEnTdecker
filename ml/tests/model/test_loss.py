import torch
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

    density = (0.9 / (0.1 * torch.sqrt(torch.tensor(2 * torch.pi)))
               + 0.1 * torch.exp(torch.tensor(-12.5)) / torch.sqrt(torch.tensor(2 * torch.pi)))
    torch.testing.assert_close(loss, -torch.log(density), atol=1e-6, rtol=1e-6)


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

