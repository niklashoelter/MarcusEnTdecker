import torch
import torch.nn as nn
from ml.model.ffn.mixture_density_network import MDN
from ml.model.gnn.encoders import *
from torch_geometric.nn import global_add_pool


class GraphMDN(nn.Module):
    """
    Graph encoder with Mixture Density Network heads (graph- and/or node-level).
    """

    def __init__(self,
                 encoder_type='dmpnn',
                 node_hidden_dim=128,
                 encoder_layers=3,
                 encoder_dropout=0.1,
                 num_graph_tasks=1,
                 num_node_tasks=0,
                 weight_graph_loss=0.5,
                 n_components=3,
                 mdn_hidden_dims=[128, 256, 128],
                 mdn_dropout=0.1,
                 mdn_activation='relu',
                 use_batch_norm=True,
                 epsilon=1e-6,
                 lambda_cal=0.0,
                 lambda_sharp=0.0,
                 device='auto',
                 **encoder_args):
        """
        Initialize the model.

        Args:
            encoder_type (str): Type of encoder to use ('dmpnn', 'gat', or 'mpnn').
            node_hidden_dim (int): Hidden dimension for node embeddings.
            encoder_layers (int): Number of layers in the encoder.
            encoder_dropout (float): Dropout rate for the encoder.
            num_graph_tasks (int): Number of graph-level output tasks.
            weight_graph_loss (float): Weight of graph loss. Should be between 0 and 1. Weight of node loss will be (1 - weight_graph_loss). Default is 50/50
            num_node_tasks (int): Number of atom-level output tasks.
            n_components (int): Number of Gaussian components in MDN.
            mdn_hidden_dims (list): Hidden layer dimensions for MDN.
            mdn_dropout (float): Dropout rate for MDN.
            mdn_activation (str): Activation function for MDN.
            use_batch_norm (bool): Whether to use batch normalization in MDN.
            epsilon (float): Small constant for numerical stability.
            lambda_cal (float): Weight for calibration loss (encourages |error| ≈ uncertainty). Default 0.0 (disabled).
            lambda_sharp (float): Weight for sharpness penalty (penalizes overconfident predictions). Default 0.0 (disabled).
            device (str): Device to run the model on.
            **encoder_args: Additional arguments for the encoder (e.g., num_heads for GAT, walk_length for positional encoding).
        """
        super().__init__()

        # Device handling
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif 'cuda' in device and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        self.device = torch.device(device)

        if num_graph_tasks == 0 and num_node_tasks == 0:
            raise ValueError("At least one of num_graph_tasks or num_node_tasks must be greater than 0.")

        if not (0 <= weight_graph_loss <= 1):
            raise ValueError("weight_graph_loss must be between 0 and 1.")

        # Store hyperparameters
        self.encoder_type = encoder_type
        self.node_hidden_dim = node_hidden_dim
        self.encoder_layers = encoder_layers
        self.encoder_dropout = encoder_dropout
        self.encoder_args = encoder_args
        self.num_graph_tasks = num_graph_tasks
        self.num_node_tasks = num_node_tasks
        self.weight_graph_loss = weight_graph_loss
        self.n_components = n_components
        self.mdn_hidden_dims = mdn_hidden_dims
        self.mdn_dropout = mdn_dropout
        self.mdn_activation = mdn_activation
        self.use_batch_norm = use_batch_norm
        self.epsilon = epsilon
        self.lambda_cal = lambda_cal
        self.lambda_sharp = lambda_sharp

        # Layers (initialized lazily)
        self.encoder = None
        self.mdn_graph = None
        self.mdn_node = None
        self._initialized = False

    def _initialize(self, node_input_dim, edge_input_dim):
        """Initialize layers with actual input dimensions."""
        encoder_classes = {
            'dmpnn': DMPNN,
            'gat': GAT,
            'mpnn': MPNN,
            'gin': GIN,
            'base_transformer': BaseGraphTransformer,
            'transformer_v2': GraphTransformer, # alias of 'transformer'
            'transformer': GraphTransformer,
            'gps': GraphGPS
        }
        encoder_class = encoder_classes.get(self.encoder_type.lower())
        if not encoder_class:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        self.encoder = encoder_class(
            node_features=node_input_dim,
            edge_features=edge_input_dim,
            hidden_size=self.node_hidden_dim,
            depth=self.encoder_layers,
            dropout=self.encoder_dropout,
            **self.encoder_args
        ).to(self.device)

        if self.num_graph_tasks > 0:
            self.mdn_graph = MDN(
                input_dim=self.node_hidden_dim,
                output_dim=self.num_graph_tasks,
                hidden_dims=self.mdn_hidden_dims,
                n_components=self.n_components,
                dropout=self.mdn_dropout,
                activation=self.mdn_activation,
                use_batch_norm=self.use_batch_norm,
                epsilon=self.epsilon
            ).to(self.device)

        if self.num_node_tasks > 0:
            self.mdn_node = MDN(
                input_dim=self.node_hidden_dim,
                output_dim=self.num_node_tasks,
                hidden_dims=self.mdn_hidden_dims,
                n_components=self.n_components,
                dropout=self.mdn_dropout,
                activation=self.mdn_activation,
                use_batch_norm=self.use_batch_norm,
                epsilon=self.epsilon
            ).to(self.device)

        self._initialized = True

    def forward(self, data):
        """Forward pass through the model."""
        data = data.to(self.device)
        if not self._initialized:
            node_input_dim = data.x.size(1)
            edge_input_dim = data.edge_attr.size(1) if data.edge_attr is not None else 0
            self._initialize(node_input_dim, edge_input_dim)

        # Node embeddings; graph head uses their sum pool

        node_embeddings = self.encoder(data, return_node_embeddings=True)

        out_graph = None
        out_node = None

        if self.num_graph_tasks > 0:
            # Pool
            graph_embedding = global_add_pool(node_embeddings, data.batch)
            out_graph = self.mdn_graph(graph_embedding)

        if self.num_node_tasks > 0:
            out_node = self.mdn_node(node_embeddings)

        # Return logic
        if self.num_graph_tasks > 0 and self.num_node_tasks > 0:
            return out_graph, out_node
        elif self.num_graph_tasks > 0:
            return out_graph
        elif self.num_node_tasks > 0:
            return out_node
        else:
             raise RuntimeError("No tasks configured.")

    def get_config(self):
        """Get model configuration as a dictionary."""
        config = {
            'encoder_type': self.encoder_type,
            'node_hidden_dim': self.node_hidden_dim,
            'encoder_layers': self.encoder_layers,
            'encoder_dropout': self.encoder_dropout,
            'num_graph_tasks': self.num_graph_tasks,
            'num_node_tasks': self.num_node_tasks,
            'n_components': self.n_components,
            'mdn_hidden_dims': self.mdn_hidden_dims,
            'mdn_dropout': self.mdn_dropout,
            'mdn_activation': self.mdn_activation,
            'use_batch_norm': self.use_batch_norm,
            'epsilon': self.epsilon,
            'lambda_cal': self.lambda_cal,
            'lambda_sharp': self.lambda_sharp,
            'device': str(self.device),
            'atom_wise': self.atom_wise
        }
        config.update(self.encoder_args)
        return config

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  encoder_type='{self.encoder_type}',\n"
            f"  node_hidden_dim={self.node_hidden_dim},\n"
            f"  encoder_layers={self.encoder_layers},\n"
            f"  num_graph_tasks={self.num_graph_tasks},\n"
            f"  num_node_tasks={self.num_node_tasks},\n"
            f"  n_components={self.n_components},\n"
            f"  device={self.device},\n"
            f"  initialized={self._initialized}\n"
            f")"
        )
