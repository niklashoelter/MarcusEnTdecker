import torch.nn as nn
import torch.nn.functional as F

class MDN(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 n_components=3,
                 hidden_dims=[256, 128],
                 dropout=0.1,
                 activation='relu',
                 use_batch_norm=True,
                 epsilon=1e-6):
        super().__init__()
        self.output_dim = output_dim
        self.n_components = n_components
        self.epsilon = epsilon

        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build hidden layers dynamically
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(self.activation)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers) if layers else nn.Identity()

        self.mu_layer = nn.Linear(prev_dim, output_dim * n_components)
        self.sigma_layer = nn.Linear(prev_dim, output_dim * n_components)
        self.pi_layer = nn.Linear(prev_dim, output_dim * n_components)

        # Initialize weights
        nn.init.constant_(self.pi_layer.bias, 0)
        nn.init.xavier_uniform_(self.pi_layer.weight)

    def forward(self, x):
        h = self.hidden_layers(x)

        mu = self.mu_layer(h).view(-1, self.output_dim, self.n_components)

        log_sigma = self.sigma_layer(h).view(-1, self.output_dim, self.n_components)
        sigma = F.softplus(log_sigma) + self.epsilon

        pi_logits = self.pi_layer(h).view(-1, self.output_dim, self.n_components)
        pi = F.softmax(pi_logits, dim=-1)

        return mu, sigma, pi