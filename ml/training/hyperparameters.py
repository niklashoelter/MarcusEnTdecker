"""
The two hyperparameter sets of the graph MDN, by name.

Both come from Leon Schlosser's Optuna search, whose trials are in
ml/results/hyperparameter_optimization/optuna_results.csv.

- `trial_0`: trial 0 of that search, with the values exactly as his training script
  (og_data_opt_param/train.py) carried them, some rounded: encoder and MDN dropout to nine
  digits, the weight decay to 1.23e-5. These made ml/data/model_final.pt and the other
  checkpoints in ml/results/trial_0/. He used trial 0 by mistake.
- `trial_55`: trial 55, the trial with the lowest validation loss, at the precision Optuna
  recorded.

Everything else about training is the same for both sets and lives in train.py.
"""

HYPERPARAMETERS = {
    "trial_0": {
        "node_hidden_dim": 512,
        "encoder_layers": 5,
        "encoder_dropout": 0.222308817,
        "num_heads": 4,
        "walk_length": 5,
        "pe_dim": 64,
        "n_components": 5,
        "mdn_hidden_dims": [128, 128],
        "mdn_dropout": 0.173327823,
        "lr": 0.0000898964645635793,
        "weight_decay": 1.23E-05,
    },
    "trial_55": {
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
    },
}

#: The keys of a checkpoint's `model_config` that tell the sets apart.
ARCHITECTURE_KEYS = ("node_hidden_dim", "encoder_layers", "encoder_dropout", "num_heads",
                     "walk_length", "pe_dim", "n_components", "mdn_hidden_dims", "mdn_dropout")


def identify(model_config):
    """The name of the set a checkpoint's `model_config` was trained with, or None."""
    for name, params in HYPERPARAMETERS.items():
        if all(model_config.get(k) == params[k] for k in ARCHITECTURE_KEYS):
            return name
    return None
