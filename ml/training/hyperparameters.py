"""
Hyperparameters of the graph MDN: trial 55 (lowest validation loss) of Leon Schlosser's
Optuna search, ml/results/hyperparameter_optimization/optuna_results.csv.
"""

#: Recorded in checkpoints under `hyperparameters`.
NAME = "trial_55"

HYPERPARAMETERS = {
    "node_hidden_dim": 512,
    "encoder_layers": 5,
    "encoder_dropout": 0.07465492107616403,
    "num_heads": 4,
    "walk_length": 7,
    "pe_dim": 64,
    "n_components": 5,
    "mdn_hidden_dims": [128, 128],
    "mdn_dropout": 0.15748581493970132,
    "lr": 0.0002432899422488752,
    "weight_decay": 1.6542298899097116e-06,
}
