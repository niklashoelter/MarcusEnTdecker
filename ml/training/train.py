"""
Train the graph MDN with one hyperparameter set, on a split or on all data.

Reads   ml/data/training_dataset_full.csv, ml/data/splits/<split>.csv
Writes  ml/results/<set>/<split>/model.pt                     the checkpoint with the lowest
                                                              validation loss
        ml/results/<set>/<split>/test_predictions_graph.csv   its test-set predictions,
        ml/results/<set>/<split>/test_predictions_node.csv    as test_predictions.py writes them
        ml/results/<set>/full/model.pt                        with `full`: trained on all data

    uv run --extra ml python -m ml.training.train trial_55 random_101 [random_143 ...] [--device cuda:0]
    uv run --extra ml python -m ml.training.train trial_55 full [--device cuda:0]

<set> is a name from hyperparameters.py. A finished model.pt is never overwritten; while a
run is going, its best checkpoint is kept as model.pt.partial and renamed when it ends.

Leon Schlosser's training script for the model in ml/data/model_final.pt,
playground/model_setup/save_model_and_inspect/new_data/graph_gps/og_data_opt_param/train.py.
Its settings, model, optimizer, learning-rate schedule, early stopping and checkpoint are
unchanged. What differs: the hyperparameters are a named set instead of constants; the data
come from ml/data/training_dataset_full.csv, the output of his `DataLoader.compute_targets` on
the original dataset, instead of recomputing it from the raw energies; the split comes from
ml/data/splits/ instead of being drawn here (the same draw, see splits.py); the plots and the
copies of the split data are not written; the final test predictions are written by
test_predictions.py from the saved best checkpoint, which is the model his script reloads for
that step; and the checkpoint also records its set, split and seed.

Added for this repository, not in his script:
- Seeding. Python, numpy and torch are seeded with the split's seed (42 for the out-of-sample
  split and for `full`). GPU kernels are not deterministic, so a rerun is close, not identical.
- `full`: trains on every molecule. With no validation set there is no early stopping; the
  run lasts as many epochs as the median best epoch of the set's five random-split runs, with
  the same warmup and a cosine schedule over that length, and keeps the last epoch.
"""

import argparse
import random
import time

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ml.model.graph_mdn import GraphMDN
from ml.model.inference import predict
from ml.training.hyperparameters import HYPERPARAMETERS
from ml.training.loop import train_epoch, validate_epoch
from ml.training.splits import OOS_SEED, RANDOM_SEEDS, read_split
from ml.training.test_predictions import (RESULTS, load_dataset, run_dir,
                                          write_test_predictions)
from ml.utils.molecule_dataset import create_dataloaders

# ------------------------------CONFIG------------------------------
EPOCHS = 200
BATCH_SIZE = 32

# Early stopping
EARLY_STOPPING_PATIENCE = 20  # epochs without improvement before training stops
EARLY_STOPPING_MIN_DELTA = 1e-4  # smallest improvement that counts

MPNN_ENCODER = 'gps'
WARMUP_EPOCHS = 12
INFERENCE_MODE = 'mixture_mean'

GRAPH_TARGETS = ['adiabatic_energy', 'lambda_acceptor_singlet', 'lambda_donor_triplet']
NODE_TARGETS = ['spin_population']
ALL_TARGETS = GRAPH_TARGETS + NODE_TARGETS

FULL = 'full'
FULL_SEED = 42


def split_seed(split):
    if split.startswith('random_'):
        return int(split.split('_')[1])
    if split == 'oos_styryl':
        return OOS_SEED
    if split == FULL:
        return FULL_SEED
    raise ValueError(f"unknown split: {split}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_metrics(result):
    for task in ('graph', 'node'):
        targets, preds = result[task]['targets'], result[task]['predictions']
        print("MAE: ", mean_absolute_error(targets, preds, multioutput='raw_values'))
        print("RMSE: ", root_mean_squared_error(targets, preds, multioutput='raw_values'))
        print("R2: ", r2_score(targets, preds, multioutput='raw_values'))


def build_model(params, device, train_loader):
    encoder_args = {'num_heads': params['num_heads'],
                    'walk_length': params['walk_length'],
                    'pe_dim': params['pe_dim']}

    model = GraphMDN(
        encoder_type=MPNN_ENCODER,
        node_hidden_dim=params['node_hidden_dim'],
        encoder_layers=params['encoder_layers'],
        encoder_dropout=params['encoder_dropout'],
        num_graph_tasks=len(GRAPH_TARGETS),
        num_node_tasks=len(NODE_TARGETS),
        n_components=params['n_components'],
        mdn_hidden_dims=params['mdn_hidden_dims'],
        mdn_dropout=params['mdn_dropout'],
        mdn_activation='relu',
        use_batch_norm=True,
        epsilon=1e-6,
        device=device,
        **encoder_args
    )

    # Initialize the lazy modules
    sample_batch = next(iter(train_loader))
    sample_batch = sample_batch.to(device)
    model(sample_batch)

    model_config = {
        'encoder_type': MPNN_ENCODER,
        'node_hidden_dim': params['node_hidden_dim'],
        'encoder_layers': params['encoder_layers'],
        'encoder_dropout': params['encoder_dropout'],
        'num_graph_tasks': len(GRAPH_TARGETS),
        'num_node_tasks': len(NODE_TARGETS),
        'n_components': params['n_components'],
        'mdn_hidden_dims': params['mdn_hidden_dims'],
        'mdn_dropout': params['mdn_dropout'],
        'mdn_activation': 'relu',
        'use_batch_norm': True,
        'epsilon': 1e-6,
        **encoder_args
    }
    return model, model_config


def optimizer_and_scheduler(model, params, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'],
                                  weight_decay=params['weight_decay'])
    scheduler1 = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=epochs - WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[WARMUP_EPOCHS])
    return optimizer, scheduler


def checkpoint_dict(model, scaler, epoch, val_loss, model_config, name, split, seed):
    return {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'epoch': epoch,
        'val_loss': val_loss,
        'target_columns_graph': GRAPH_TARGETS,
        'target_columns_node': None,
        'model_config': model_config,
        'hyperparameters': name,
        'split': split,
        'seed': seed,
    }


def train(name, split, device, data=None, epochs=EPOCHS):
    """Train set `name` on `split` with early stopping, then write its test predictions."""
    params = HYPERPARAMETERS[name]
    out_dir = run_dir(name, split)
    checkpoint_path = out_dir / 'model.pt'
    data = load_dataset() if data is None else data

    if checkpoint_path.exists():
        print(f"{checkpoint_path} exists; not retraining")
    else:
        seed = split_seed(split)
        seed_everything(seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        partial_path = out_dir / 'model.pt.partial'

        # -------------------------------PREPARE DATA-----------------------------------
        train_idx, val_idx, test_idx = read_split(split)
        smiles = data['smiles'].values

        train_loader, validation_loader, test_loader, scaler = create_dataloaders(
            smiles, train_idx, val_idx, test_idx,
            batch_size=BATCH_SIZE,
            num_workers=0,
            scaler_path=None,
            node_targets=data[NODE_TARGETS].values,
            graph_targets=data[GRAPH_TARGETS].values,
            positional_encoding_kwargs={'walk_length': params['walk_length']}
        )

        # -------------------------------TRAINING LOOP--------------------------------
        print(f"Start training {name} on {split} (seed {seed})")
        model, model_config = build_model(params, device, train_loader)
        optimizer, scheduler = optimizer_and_scheduler(model, params, epochs)

        best_val_loss = float('inf')
        no_improvement_epochs = 0

        for epoch in range(epochs):
            epoch_start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, device)
            loss_dict = validate_epoch(model, validation_loader, device)
            val_loss, graph_loss, node_loss = loss_dict['loss'], loss_dict['loss_g'], loss_dict['loss_n']
            scheduler.step()
            epoch_end = time.time()
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Graph Loss: {graph_loss:.4f} | "
                  f"Node Loss: {node_loss:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Epoch Time: {epoch_end - epoch_start:.2f} s", flush=True)

            if epoch % 25 == 0 and epoch > 0:
                print(f"Evaluating on test set in epoch {epoch + 1} on {split}...")
                print_metrics(predict(model, test_loader, device, scaler, mode=INFERENCE_MODE))

            # Early stopping check
            if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
                best_val_loss = val_loss
                no_improvement_epochs = 0
                if epoch > epochs * 0.2:
                    torch.save(checkpoint_dict(model, scaler, epoch + 1, val_loss, model_config,
                                               name, split, seed), partial_path)
                    print(f"New best model saved with val loss: {best_val_loss:.4f} at epoch {epoch + 1}")
            else:
                no_improvement_epochs += 1

            if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered. No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
                break

        if not partial_path.exists():
            raise RuntimeError(f"{name} {split}: no checkpoint was saved (the best epoch "
                               f"came before epoch {int(epochs * 0.2) + 1})")
        partial_path.replace(checkpoint_path)

        # -------------------------------FINAL TEST EVALUATION--------------------------------
        print("\nFinal evaluation on test set...")
        best_checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        model.eval()
        print_metrics(predict(model, test_loader, device, scaler, mode=INFERENCE_MODE))

    write_test_predictions(name, split, device=device, data=data)
    print(f"Test predictions saved to {out_dir}")


def full_epochs(name):
    """The median best epoch of the set's five random-split runs."""
    epochs = []
    for seed in RANDOM_SEEDS:
        path = run_dir(name, f"random_{seed}") / 'model.pt'
        if not path.exists():
            raise FileNotFoundError(f"{path} is needed to fix the length of the full run")
        epochs.append(torch.load(path, map_location='cpu', weights_only=False)['epoch'])
    return int(np.median(epochs))


def train_full(name, device, data=None, epochs=None):
    """Train set `name` on every molecule for a fixed number of epochs; keep the last."""
    params = HYPERPARAMETERS[name]
    out_dir = run_dir(name, FULL)
    checkpoint_path = out_dir / 'model.pt'
    if checkpoint_path.exists():
        print(f"{checkpoint_path} exists; not retraining")
        return
    epochs = full_epochs(name) if epochs is None else epochs
    if epochs <= WARMUP_EPOCHS:
        raise ValueError(f"{epochs} epochs do not outlast the {WARMUP_EPOCHS}-epoch warmup")

    data = load_dataset() if data is None else data
    seed_everything(FULL_SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_idx = np.arange(len(data))
    # create_dataloaders needs validation and test indices; their loaders are not used.
    unused = all_idx[:BATCH_SIZE]
    train_loader, _, _, scaler = create_dataloaders(
        data['smiles'].values, all_idx, unused, unused,
        batch_size=BATCH_SIZE,
        num_workers=0,
        scaler_path=None,
        node_targets=data[NODE_TARGETS].values,
        graph_targets=data[GRAPH_TARGETS].values,
        positional_encoding_kwargs={'walk_length': params['walk_length']}
    )

    print(f"Start training {name} on all {len(all_idx)} molecules for {epochs} epochs")
    model, model_config = build_model(params, device, train_loader)
    optimizer, scheduler = optimizer_and_scheduler(model, params, epochs)

    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Epoch Time: {time.time() - epoch_start:.2f} s", flush=True)

    partial_path = out_dir / 'model.pt.partial'
    torch.save(checkpoint_dict(model, scaler, epochs, float('nan'), model_config,
                               name, FULL, FULL_SEED), partial_path)
    partial_path.replace(checkpoint_path)
    print(f"Saved {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("hyperparameters", choices=sorted(HYPERPARAMETERS))
    parser.add_argument("splits", nargs="+", help="split names, e.g. random_101 oos_styryl, or full")
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    data = load_dataset()
    for split in args.splits:
        if split == FULL:
            train_full(args.hyperparameters, args.device, data)
        else:
            train(args.hyperparameters, split, args.device, data)


if __name__ == "__main__":
    main()
