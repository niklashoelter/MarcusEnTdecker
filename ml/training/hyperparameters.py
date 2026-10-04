"""
The hyperparameters of the graph MDN.

They are trial 55 of Leon Schlosser's Optuna search, the trial with the lowest validation
loss, at the precision Optuna recorded; all trials are in
ml/results/hyperparameter_optimization/optuna_results.csv. Everything else about training
lives in train.py.
"""

#: The name the checkpoints record under `hyperparameters`.
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
