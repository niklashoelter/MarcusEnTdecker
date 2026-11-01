import torch
from torch_geometric.utils import scatter
from ml.model.loss import (
    mdn_loss,
    mdn_loss_per_sample,
    calibration_loss,
    calibration_loss_per_sample,
    sharpness_penalty,
    sharpness_penalty_per_sample,
)

def train_epoch(model, loader, optimizer, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        outputs = model(batch)

        loss_dict = compute_loss(model, outputs, batch)
        loss = loss_dict["loss"]

        if torch.isnan(loss):
            raise ValueError("NaN loss detected!")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def validate_epoch(model, loader, device):
    """Validate and return average loss."""
    model.eval()
    totals = {}

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            outputs = model(batch)
            loss_dict = compute_loss(model, outputs, batch)
            n = batch.num_graphs
            for key, value in loss_dict.items():
                totals[key] = totals.get(key, 0.0) + value.item() * n

    n_graphs = len(loader.dataset)
    return {key: total / n_graphs for key, total in totals.items()}


def _compute_uncertainty_losses(mu, sigma, pi, target, batch_index=None):
    """
    Compute calibration and sharpness losses for a given MDN output.

    Args:
        mu, sigma, pi: MDN outputs
        target: Target values
        batch_index: If provided, aggregate per-sample losses per graph (for node-level).
                     If None, compute global mean (for graph-level).

    Returns:
        loss_cal: Calibration loss scalar
        loss_sharp: Sharpness penalty scalar
    """
    if batch_index is not None:
        # Node-level: per-sample losses, then mean over nodes per graph, then mean over graphs
        cal_per_sample = calibration_loss_per_sample(mu, sigma, pi, target)
        sharp_per_sample = sharpness_penalty_per_sample(mu, sigma, pi, target)
        loss_cal = scatter(cal_per_sample, batch_index, dim=0, reduce="mean").mean()
        loss_sharp = scatter(sharp_per_sample, batch_index, dim=0, reduce="mean").mean()
    else:
        # Graph-level: global mean
        loss_cal = calibration_loss(mu, sigma, pi, target)
        loss_sharp = sharpness_penalty(mu, sigma, pi, target)

    return loss_cal, loss_sharp


def compute_loss(model, outputs, batch):

    lambda_cal = getattr(model, 'lambda_cal', 0.0)
    lambda_sharp = getattr(model, 'lambda_sharp', 0.0)

    if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[0], tuple) and isinstance(outputs[1], tuple):
        out_graph, out_node = outputs

        # Graph Loss (using batch.y)
        mu_g, sigma_g, pi_g = out_graph
        loss_g = mdn_loss_per_sample(mu_g, sigma_g, pi_g, batch.y).mean()  # Mean over graph samples

        # Node Loss (using batch.node_y) returning mean over nodes per graph, then mean over graphs
        mu_n, sigma_n, pi_n = out_node
        node_losses = mdn_loss_per_sample(mu_n, sigma_n, pi_n, batch.node_y)
        loss_n = scatter(node_losses, batch.batch, dim=0, reduce="mean").mean()

        #  weighting
        weight_g = model.weight_graph_loss
        weight_n = 1 - model.weight_graph_loss

        loss = (weight_g * loss_g +
                weight_n * loss_n)

        result = {"loss": loss, "pure_loss": loss, "loss_g": loss_g, "loss_n": loss_n}

        # Uncertainty calibration losses
        cal_g, sharp_g = _compute_uncertainty_losses(mu_g, sigma_g, pi_g, batch.y)
        cal_n, sharp_n = _compute_uncertainty_losses(mu_n, sigma_n, pi_n, batch.node_y, batch.batch)

        loss_cal = weight_g * cal_g + weight_n * cal_n
        loss_sharp = weight_g * sharp_g + weight_n * sharp_n

        result["loss_cal"] = loss_cal
        result["loss_sharp"] = loss_sharp

        if lambda_cal > 0 or lambda_sharp > 0:
            result["loss"] = loss + lambda_cal * loss_cal + lambda_sharp * loss_sharp

        return result

    else:
        # single output type: no weighting
        if model.num_graph_tasks > 0 and model.num_node_tasks == 0:
            mu, sigma, pi = outputs
            loss = mdn_loss(mu, sigma, pi, batch.y)  # equivalent to mdn_loss_per_sample(...).mean() for graph-level loss
            result = {"loss": loss, "pure_loss": loss}

            loss_cal, loss_sharp = _compute_uncertainty_losses(mu, sigma, pi, batch.y)
            result["loss_cal"] = loss_cal
            result["loss_sharp"] = loss_sharp

            if lambda_cal > 0 or lambda_sharp > 0:
                result["loss"] = loss + lambda_cal * loss_cal + lambda_sharp * loss_sharp

            return result

        elif model.num_node_tasks > 0 and model.num_graph_tasks == 0:
            mu, sigma, pi = outputs
            node_losses = mdn_loss_per_sample(mu, sigma, pi, batch.node_y)
            loss = scatter(node_losses, batch.batch, dim=0, reduce="mean").mean()  # mean over nodes per graph, then mean over graphs
            result = {"loss": loss, "pure_loss": loss}

            loss_cal, loss_sharp = _compute_uncertainty_losses(mu, sigma, pi, batch.node_y, batch.batch)
            result["loss_cal"] = loss_cal
            result["loss_sharp"] = loss_sharp

            if lambda_cal > 0 or lambda_sharp > 0:
                result["loss"] = loss + lambda_cal * loss_cal + lambda_sharp * loss_sharp

            return result

        else:
            raise ValueError("Model output format not recognized. Expected either graph-level or node-level outputs, or a tuple of both.")