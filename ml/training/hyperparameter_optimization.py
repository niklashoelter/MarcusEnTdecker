"""
Hyperparameter optimization of the graph MDN with Optuna.

Reads   ml/data/training_dataset_full.csv, ml/data/splits/random_42.csv
Writes  ml/results/hyperparameter_optimization/optuna_results.csv   one row per trial

    uv sync --extra ml
    uv run --extra ml python -m ml.training.hyperparameter_optimization

Objective: lowest validation loss of a trial; median pruner. Running it overwrites
optuna_results.csv, whose trial 55 hyperparameters.py holds.

Adapted from Leon Schlosser's hyperparameter-optimization script.
"""

from pathlib import Path

import optuna
import torch
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ml.model.graph_mdn import GraphMDN
from ml.model.inference import predict
from ml.training.loop import train_epoch, validate_epoch
from ml.training.splits import read_split
from ml.training.test_predictions import RESULTS, load_dataset
from ml.utils.molecule_dataset import create_dataloaders

# ------------------------------CONFIG------------------------------
N_TRIALS = 75
EPOCHS_PER_TRIAL = 125
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
SPLIT = 'random_42'

save_path = RESULTS / 'hyperparameter_optimization'

GRAPH_TARGETS = ['adiabatic_energy', 'lambda_acceptor_singlet', 'lambda_donor_triplet']
NODE_TARGETS = ['spin_population']

data = load_dataset()
train_idx, val_idx, test_idx = read_split(SPLIT)

smiles = data['smiles'].values
graph_targets = data[GRAPH_TARGETS].values
node_targets = data[NODE_TARGETS].values


def objective(trial):
    # ------------------ Hyperparameters to Optimize ------------------

    # 1. Transformer Encoder Hyperparameters
    # hidden_dim must be divisible by num_heads
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    base_hidden_dim = trial.suggest_categorical('base_node_hidden_dim', [32, 64, 128])
    # Ensure divisible by heads by multiplying
    node_hidden_dim = base_hidden_dim * num_heads

    walk_length = trial.suggest_int('walk_length', 4, 20)
    pe_dim = trial.suggest_categorical('pe_dim', [16, 32, 64])

    encoder_layers = trial.suggest_int('encoder_layers', 2, 6)
    encoder_dropout = trial.suggest_float('encoder_dropout', 0.0, 0.5)

    # 2. MDN Hyperparameters
    mdn_dropout = trial.suggest_float('mdn_dropout', 0.0, 0.5)

    # MDN Hidden Dimensions (Number of layers and size)
    n_mdn_layers = trial.suggest_int('n_mdn_layers', 1, 3)
    mdn_hidden_dim_size = trial.suggest_categorical('mdn_hidden_dim_size', [64, 128, 256, 512])
    mdn_hidden_dims = [mdn_hidden_dim_size] * n_mdn_layers

    n_components = trial.suggest_int('n_components', 2, 5)

    # 3. Optimization Hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # ------------------ Model Setup ------------------
    try:

        train_loader, validation_loader, test_loader, scaler = create_dataloaders(
            smiles, train_idx, val_idx, test_idx,
            graph_targets=graph_targets,
            node_targets=node_targets,
            batch_size=BATCH_SIZE,
            num_workers=0,
            scaler_path=None,
            positional_encoding_kwargs={'walk_length': walk_length}
        )

        model = GraphMDN(
            encoder_type='gps',
            node_hidden_dim=node_hidden_dim,
            encoder_layers=encoder_layers,
            encoder_dropout=encoder_dropout,
            num_graph_tasks=3,
            num_node_tasks=1,
            n_components=n_components,
            mdn_hidden_dims=mdn_hidden_dims,
            mdn_dropout=mdn_dropout,
            mdn_activation='relu',
            use_batch_norm=True,
            epsilon=1e-6,
            device=DEVICE,
            # Transformer specific args
            num_heads=num_heads,
            walk_length=walk_length,
            pe_dim=pe_dim
        )

        # Initialize lazy modules
        sample_batch = next(iter(train_loader))
        sample_batch = sample_batch.to(DEVICE)
        model(sample_batch)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        warmup_epochs = int(12)
        scheduler1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        scheduler2 = CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_TRIAL - warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])

        # ------------------ Training Loop ------------------
        best_val_loss = float('inf')

        print("Starting trial with hyperparameters: ")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")

        for epoch in range(EPOCHS_PER_TRIAL):
            train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
            val_loss_dict = validate_epoch(model, validation_loader, DEVICE)
            val_loss, graph_loss, node_loss = val_loss_dict['loss'], val_loss_dict['loss_g'], val_loss_dict['loss_n']
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch + 1}/{EPOCHS_PER_TRIAL} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Graph Loss: {graph_loss:.4f} | "
                  f"Node Loss: {node_loss:.4f} | "
                  f"LR: {current_lr:.6f} | ")

            # Report intermediate objective value
            trial.report(val_loss, epoch)

            # Handle pruning (stop unpromising trials early)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        print("Final evaluation on test set:")
        test_result = predict(model, test_loader, DEVICE, scaler)
        for task in ('graph', 'node'):
            targets, preds = test_result[task]['targets'], test_result[task]['predictions']
            print("MAE: ", mean_absolute_error(targets, preds, multioutput='raw_values'))
            print("RMSE: ", root_mean_squared_error(targets, preds, multioutput='raw_values'))
            print("R2: ", r2_score(targets, preds, multioutput='raw_values'))

        return best_val_loss

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"Trial failed with OOM: {e}")
            torch.cuda.empty_cache()
            return float('inf')  # Fail gracefully
        else:
            raise e


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__.splitlines()[1]).parse_args()
    print(f"Starting Optuna Optimization on {DEVICE}")

    # Create study
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "=" * 50)
    print("OPTUNA RESULTS")
    print("=" * 50)

    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print(f"\nBest Validation Loss: {study.best_value}")

    # Save results to CSV
    df = study.trials_dataframe()
    results_path = Path(save_path) / 'optuna_results.csv'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False, lineterminator="\n")
    print(f"\nResults saved to {results_path}")
