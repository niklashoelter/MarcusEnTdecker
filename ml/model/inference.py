# Aliases the pre-rename module paths so the pickled DualScaler in model_final.pt resolves.
from ml.model import checkpoint as _checkpoint  # noqa: F401

import numpy as np
import torch
from pathlib import Path
from ml.model.graph_mdn import GraphMDN
from torch.utils.data import DataLoader
from ml.utils.molecule_dataset import MoleculeDataset, collate_fn
from ml.model.ffn.mdn_utils import compute_mixture_mean, compute_max_mode

def predict(model, loader, device, scaler, mode='mixture_mean'):
    '''
    Make predictions with the MDN model. Supports dual-head (Graph + Node) models.
    Returns structured results as a dictionary.

    Args:
        model: The MDN model to use for predictions.
        loader: DataLoader containing the data to predict on.
        device: Device to run predictions on.
        scaler: Scaler for inverse transforming predictions.
        mode: Prediction mode - 'mixture_mean' (default) or 'max_mode'
              - 'mixture_mean': Weighted average of all modes
              - 'max_mode': Use the mode with highest weight

    Returns:
        dict: A dictionary with 'graph' and 'node' keys, each containing:
            'predictions', 'uncertainties', 'weights', 'targets', 'mdn_params'
    '''
    model.eval()

    g_data = {'preds': [], 'targets': [], 'uncertainty': [], 'weights': [], 'mu': [], 'sigma': [], 'pi': []}
    n_data = {'preds': [], 'targets': [], 'uncertainty': [], 'weights': [], 'mu': [], 'sigma': [], 'pi': []}

    dual_mode = False
    node_only_mode = False

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)

            if isinstance(outputs, tuple) and len(outputs) == 2:
                dual_mode = True
                # Process graph output
                preds, targets, unc, weights, mdn = _process_mdn_output(*outputs[0], batch, 'y', mode=mode)
                g_data['preds'].append(preds)
                g_data['targets'].append(targets)
                g_data['uncertainty'].append(unc)
                g_data['weights'].append(weights)
                g_data['mu'].append(mdn['mu'])
                g_data['sigma'].append(mdn['sigma'])
                g_data['pi'].append(mdn['pi'])

                # Process node output
                preds, targets, unc, weights, mdn = _process_mdn_output(*outputs[1], batch, 'node_y', mode=mode)
                n_data['preds'].append(preds)
                n_data['targets'].append(targets)
                n_data['uncertainty'].append(unc)
                n_data['weights'].append(weights)
                n_data['mu'].append(mdn['mu'])
                n_data['sigma'].append(mdn['sigma'])
                n_data['pi'].append(mdn['pi'])
            else:
                # Single head - determine if graph or node
                is_node = _is_node_output(outputs[0].shape[0], batch)
                target_attr = 'node_y' if is_node else 'y'
                data = n_data if is_node else g_data

                if is_node:
                    node_only_mode = True

                preds, targets, unc, weights, mdn = _process_mdn_output(*outputs, batch, target_attr, mode=mode)
                data['preds'].append(preds)
                data['targets'].append(targets)
                data['uncertainty'].append(unc)
                data['weights'].append(weights)
                data['mu'].append(mdn['mu'])
                data['sigma'].append(mdn['sigma'])
                data['pi'].append(mdn['pi'])

    # Get scalers
    g_scaler = scaler.graph_scaler if hasattr(scaler, 'graph_scaler') else scaler if g_data['preds'] else None
    n_scaler = scaler.node_scaler if hasattr(scaler, 'node_scaler') else scaler if n_data['preds'] else None

    # Finalize outputs
    g_result = _finalize_output(g_data, g_scaler) if g_data['preds'] else None
    n_result = _finalize_output(n_data, n_scaler) if n_data['preds'] else None

    # Clear unused heads
    if not dual_mode and not node_only_mode:
        n_result = None
    elif not dual_mode and node_only_mode:
        g_result = None

    return _build_result(g_result, n_result)

_global_model_cache = None

def build_model(checkpoint, device, smiles, batch_size=48):
    '''
    The GraphMDN a loaded checkpoint describes, with its weights in place.

    :param checkpoint: The dictionary torch.load returns for a checkpoint.
    :param device: Device to build the model on.
    :param smiles: List of SMILES whose first batch initialises the lazy layers.
    :param batch_size: Batch size of that first batch.
    '''
    model = GraphMDN(
        encoder_type=checkpoint['model_config']['encoder_type'],
        node_hidden_dim=checkpoint['model_config']['node_hidden_dim'],
        encoder_layers=checkpoint['model_config']['encoder_layers'],
        encoder_dropout=checkpoint['model_config']['encoder_dropout'],
        num_graph_tasks=checkpoint['model_config']['num_graph_tasks'],
        num_node_tasks=checkpoint['model_config']['num_node_tasks'],
        n_components=checkpoint['model_config']['n_components'],
        mdn_hidden_dims=checkpoint['model_config']['mdn_hidden_dims'],
        mdn_dropout=checkpoint['model_config']['mdn_dropout'],
        mdn_activation=checkpoint['model_config']['mdn_activation'],
        use_batch_norm=checkpoint['model_config']['use_batch_norm'],
        epsilon=checkpoint['model_config']['epsilon'],
        device=device,
        num_heads=checkpoint['model_config']['num_heads'],
        pe_dim=checkpoint['model_config']['pe_dim'],
        walk_length=checkpoint['model_config']['walk_length']
    )

    if len(smiles) == 1: #Workaround for lazy initialization
        init_smiles = list(smiles)
        init_smiles.append(smiles[0]) #Duplicate the single SMILES to create a batch of 2
    else:
        init_smiles = smiles

    init_dataset = MoleculeDataset(init_smiles, positional_encoding_kwargs=checkpoint['model_config'])
    init_loader = DataLoader(init_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    sample_batch = next(iter(init_loader))
    sample_batch = sample_batch.to(device)
    model(sample_batch)

    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_predictions(smiles: list,
                    model_path: Path = Path(
                        '/tmp/pyRates/playground/model_setup/save_model_and_inspect/new_data/graph_gps/new_loss/checkpoints/model.pt'),
                    device='cpu',
                    task: str = 'graph', #other options are node or all
                    mode: str = 'mixture_mean' #other option max_mode
                    ):
    '''
    Function to get predictions from a trained model.

    :param smiles: List of SMILES strings for which to get predictions.
    :param model_path: Path to the directory containing the trained model checkpoint (should contain 'checkpoints/model.pt').
    :param device: Device to run the model on (e.g., 'cpu' or 'cuda:0').
    :return: Tuple containing predictions, uncertainties, mode weights, and MDN parameters (mu, sigma, pi).
    :param task: Prediction task - 'graph', 'node', or 'all' (default: 'graph').
    '''

    global _global_model_cache
    BATCH_SIZE = 48
    if _global_model_cache is None:
        print(f"Loading weights into cache from {model_path}...")
        if device == 'cpu':
            checkpoint = torch.load(model_path, map_location='cpu')
        else:
            checkpoint = torch.load(str(model_path))
        print(f"Loading {len(smiles)} molecules for prediction")
        model = build_model(checkpoint, device, smiles, batch_size=BATCH_SIZE)
        scaler = checkpoint['scaler']

        _global_model_cache = (model, scaler, checkpoint)

    model, scaler, checkpoint = _global_model_cache

    test_dataset = MoleculeDataset(smiles, positional_encoding_kwargs=checkpoint['model_config'])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print("Getting predictions from ", model_path)
    print(
        f"Loaded model from epoch {checkpoint['epoch']} with val_loss {checkpoint['val_loss']:.4f} and GNN encoder type {checkpoint['model_config']['encoder_type']}")
    prediction_result = predict(model, test_loader, device, scaler, mode=mode)
    if task == 'all':
        return prediction_result

    predictions, uncertainties, mode_weights = prediction_result[task]['predictions'], prediction_result[task]['uncertainties'], prediction_result[task]['weights']
    mu, sigma, pi = prediction_result[task]['mdn_params']['mu'], prediction_result[task]['mdn_params']['sigma'], prediction_result[task]['mdn_params']['pi']

    return predictions, uncertainties, mode_weights, mu, sigma, pi


def _process_mdn_output(mu, sigma, pi, batch, target_attr, mode='mixture_mean'):
    """
    Extract predictions, uncertainties, and targets from MDN output.

    Args:
        mu: Mean values [batch_size, output_dim, n_components]
        sigma: Standard deviations [batch_size, output_dim, n_components]
        pi: Mixture weights [batch_size, output_dim, n_components]
        batch: Batch containing targets
        target_attr: Attribute name for targets in batch ('y' or 'node_y')
        mode: Computation mode - 'mixture_mean', 'max_mode', or 'each_mode'

    Returns:
        preds: Predictions as numpy array
        targets: Targets as numpy array (or None)
        unc: Uncertainties as numpy array
        weights: Mixture weights as numpy array
        mdn_params: Dictionary with mu, sigma, pi as numpy arrays
    """
    if mode == 'mixture_mean':
        mean, uncertainty = compute_mixture_mean(mu, sigma, pi)
    elif mode == 'max_mode':
        mean, uncertainty = compute_max_mode(mu, sigma, pi)
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes: 'mixture_mean', 'max_mode'")

    preds = mean.cpu().numpy()
    unc = uncertainty.cpu().numpy()
    weights = pi.cpu().numpy()

    targets = None
    if hasattr(batch, target_attr):
        raw_targets = getattr(batch, target_attr)
        if raw_targets is not None:
            if isinstance(raw_targets, torch.Tensor):
                targets = raw_targets.cpu().numpy()
            elif isinstance(raw_targets, np.ndarray):
                targets = raw_targets
            elif isinstance(raw_targets, list):
                targets = np.array(raw_targets)

            # Check if targets contains None
            if targets is not None and targets.dtype == object and (targets == None).any():
                targets = None

    mdn_params = {
        'mu': mu.cpu().numpy(),
        'sigma': sigma.cpu().numpy(),
        'pi': pi.cpu().numpy()
    }

    return preds, targets, unc, weights, mdn_params


def _stack_arrays(arrays):
    """Stack list of arrays, return None if empty or contains None."""
    if not arrays or any(a is None for a in arrays):
        return None
    return np.vstack(arrays)


def _scale_mdn_output(preds, targets, unc, mdn_params, scaler):
    """Apply inverse transform and scale MDN parameters."""
    if scaler is None:
        return preds, targets, unc, mdn_params

    preds = scaler.inverse_transform(preds)
    if targets is not None:
        targets = scaler.inverse_transform(targets)
    unc = unc * scaler.scale_

    if mdn_params is not None:
        scale = scaler.scale_[np.newaxis, :, np.newaxis]
        mean = scaler.mean_[np.newaxis, :, np.newaxis]
        mdn_params['mu'] = mdn_params['mu'] * scale + mean
        mdn_params['sigma'] = mdn_params['sigma'] * scale

    return preds, targets, unc, mdn_params


def _finalize_output(data_dict, scaler):
    """Stack and scale all arrays in data dictionary."""
    preds = _stack_arrays(data_dict['preds'])
    targets = _stack_arrays(data_dict['targets'])
    unc = _stack_arrays(data_dict['uncertainty'])
    weights = _stack_arrays(data_dict['weights'])
    mu = _stack_arrays(data_dict['mu'])
    sigma = _stack_arrays(data_dict['sigma'])
    pi = _stack_arrays(data_dict['pi'])

    mdn_params = {'mu': mu, 'sigma': sigma, 'pi': pi} if mu is not None else None

    if preds is not None:
        preds, targets, unc, mdn_params = _scale_mdn_output(preds, targets, unc, mdn_params, scaler)

    return {
        'predictions': preds,
        'uncertainties': unc,
        'weights': weights,
        'targets': targets,
        'mdn_params': mdn_params
    }


def _build_result(graph_data, node_data):
    """Build final result dictionary."""
    return {
        'graph': graph_data or {k: None for k in ['predictions', 'uncertainties', 'weights', 'targets', 'mdn_params']},
        'node': node_data or {k: None for k in ['predictions', 'uncertainties', 'weights', 'targets', 'mdn_params']}
    }


def _is_node_output(output_batch_size, batch):
    """Determine if output is node-level based on shape."""
    num_nodes = batch.x.shape[0] if hasattr(batch, 'x') else None
    num_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else None

    return (num_nodes is not None and output_batch_size == num_nodes and
            (num_graphs is None or output_batch_size != num_graphs))
