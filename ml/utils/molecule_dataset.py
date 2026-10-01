import torch
from torch.utils.data import Dataset
from ml.utils.graph_construction import smiles_to_graph
from ml.model.gnn.positional_encoding import *
from torch_geometric.data import Batch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import joblib
import numpy as np

class DualScaler:
    """Wrapper for handling separate scalers for graph and node targets."""
    def __init__(self, graph_scaler=None, node_scaler=None):
        self.graph_scaler = graph_scaler
        self.node_scaler = node_scaler

    def transform(self, targets):
        pass

    def inverse_transform(self, data):
        # Data is keys 'graph' and 'node' or tuple
        if isinstance(data, dict):
            res = {}
            if 'graph' in data and self.graph_scaler:
                res['graph'] = self.graph_scaler.inverse_transform(data['graph'])
            elif 'graph' in data:
                res['graph'] = data['graph']

            if 'node' in data and self.node_scaler:
                res['node'] = self.node_scaler.inverse_transform(data['node'])
            elif 'node' in data:
                res['node'] = data['node']
            return res
        return data

class MoleculeDataset(Dataset):

    def __init__(self, smiles_list, node_targets=None, graph_targets=None, positional_encoding_kwargs=None, **kwargs):
        self.graphs = []
        self.graph_targets = []
        self.node_targets = []

        self.has_graph_targets = graph_targets is not None
        self.has_node_targets = node_targets is not None

        for i, smiles in enumerate(smiles_list):
            graph = smiles_to_graph(smiles, use_explicit_hydrogens=False, use_stereochemistry=True)
            if graph is None:
                print(f"Warning: Invalid SMILES at idx {i}: {smiles}. Skipping.")
                continue

            # Add positional encoding
            if positional_encoding_kwargs is not None:
                walk_length = positional_encoding_kwargs['walk_length']
                graph = add_random_walk_positional_encoding(graph, walk_length=walk_length)
                graph = add_relative_random_walk_positional_encoding(graph, walk_length=walk_length)

            current_g_t = None
            current_n_t = None

            if self.has_graph_targets:
                 current_g_t = torch.tensor(graph_targets[i], dtype=torch.float)

            if self.has_node_targets:
                 n_t = node_targets[i]
                 # Handle object array wrapping
                 if isinstance(n_t, np.ndarray) and n_t.dtype == object:
                     if n_t.size == 1:
                         n_t = n_t.item()
                     else:
                         n_t = n_t.tolist()

                 # Check consistency
                 if hasattr(n_t, '__len__') and len(n_t) != graph.num_nodes:
                     print(f"Warning: Discarding molecule idx {i} (SMILES: {smiles}). "
                           f"Node count ({graph.num_nodes}) != Target length ({len(n_t)}).")
                     continue

                 current_n_t = torch.tensor(n_t, dtype=torch.float)
                 if current_n_t.ndim == 1: current_n_t = current_n_t.unsqueeze(1)

            # If validated, append
            self.graphs.append(graph)
            if self.has_graph_targets:
                self.graph_targets.append(current_g_t)
            if self.has_node_targets:
                self.node_targets.append(current_n_t)

        # Finalize storage
        if self.has_graph_targets and len(self.graph_targets) > 0:
            self.graph_targets = torch.stack(self.graph_targets)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx].clone()

        if self.has_graph_targets:
             y = self.graph_targets[idx]
             if y.ndim == 0: y = y.unsqueeze(0)
             if y.ndim == 1: y = y.unsqueeze(0)
             data.y = y

        if self.has_node_targets:
             data.node_y = self.node_targets[idx]
             # If no graph targets, use node targets as primary y (e.g. for loss functions expecting data.y)
             if not self.has_graph_targets:
                 data.y = data.node_y

        return data

def collate_fn(batch):
    # Use PyG's own batching which handles attributes
    return Batch.from_data_list(batch)

def create_dataloaders(smiles_list,
                       train_idx,
                       val_idx,
                       test_idx,
                       batch_size=64,
                       num_workers=8,
                       scaler_path=None,
                       positional_encoding_kwargs=None,
                       node_targets=None,
                       graph_targets=None,
                       **kwargs):
    '''
    Create dataloaders for training, validation, and testing. and scale targets using StandardScaler.
    '''
    if train_idx is None or val_idx is None or test_idx is None:
         raise ValueError("train_idx, val_idx, and test_idx must be provided.")

    smiles_train = smiles_list[train_idx]
    smiles_val = smiles_list[val_idx]
    smiles_test = smiles_list[test_idx]

    gt_train = None; gt_val = None; gt_test = None
    nt_train = None; nt_val = None; nt_test = None

    graph_scaler = None
    node_scaler = None
    scaler = None

    # Graph Targets
    if graph_targets is not None:
         g_train_orig = graph_targets[train_idx]
         g_val_orig = graph_targets[val_idx]
         g_test_orig = graph_targets[test_idx]

         graph_scaler = StandardScaler()
         # Reshape if needed
         if hasattr(g_train_orig, 'ndim') and g_train_orig.ndim == 1:
              g_train_orig = g_train_orig.reshape(-1, 1)
              g_val_orig = g_val_orig.reshape(-1, 1)
              g_test_orig = g_test_orig.reshape(-1, 1)

         # If list, convert to array and reshape
         elif isinstance(g_train_orig, list):
              g_train_orig = np.array(g_train_orig)
              if g_train_orig.ndim == 1: g_train_orig = g_train_orig.reshape(-1, 1)
              # Also fix val/test if list
              g_val_orig = np.array(g_val_orig)
              if g_val_orig.ndim == 1: g_val_orig = g_val_orig.reshape(-1, 1)
              g_test_orig = np.array(g_test_orig)
              if g_test_orig.ndim == 1: g_test_orig = g_test_orig.reshape(-1, 1)

         gt_train = graph_scaler.fit_transform(g_train_orig)
         gt_val = graph_scaler.transform(g_val_orig)
         gt_test = graph_scaler.transform(g_test_orig)

    # Node Targets
    if node_targets is not None:
         n_train_orig = [node_targets[i] for i in train_idx]
         n_val_orig = [node_targets[i] for i in val_idx]
         n_test_orig = [node_targets[i] for i in test_idx]

         node_scaler = StandardScaler()

         # Helper to unwrap possible object arrays
         def start_unwrap(x):
             if isinstance(x, np.ndarray) and x.dtype==object and x.size==1:
                 return x.item()
             return x

         # Collect all training node data to fit scaler
         all_train_vals = []

         # Determine if multitask based on first sample
         sample = start_unwrap(n_train_orig[0])
         is_multitask = False
         if isinstance(sample, (list, np.ndarray)):
              if len(sample) > 0 and isinstance(sample[0], (list, np.ndarray)):
                    is_multitask = True # List of lists/arrays

         if is_multitask:
              # Stack them
              for x in n_train_orig:
                  u = start_unwrap(x)
                  if isinstance(u, list): u = np.column_stack(u)
                  # if u is already array (atoms, tasks), fine.
                  all_train_vals.append(u)

              if len(all_train_vals) > 0:
                   node_scaler.fit(np.vstack(all_train_vals))
         else:
              # Single task per atom
              for x in n_train_orig:
                   u = start_unwrap(x)
                   all_train_vals.extend(np.array(u).flatten())
              node_scaler.fit(np.array(all_train_vals).reshape(-1, 1))

         # Transform
         def transform_nodes(data, s, is_mt):
             res = []
             for x in data:
                 u = start_unwrap(x)
                 if is_mt:
                     if isinstance(u, list): u = np.column_stack(u)
                     val = s.transform(u)
                     res.append(val)
                 else:
                     u_arr = np.array(u).reshape(-1, 1)
                     val = s.transform(u_arr).flatten()
                     res.append(val)
             return res

         nt_train = transform_nodes(n_train_orig, node_scaler, is_multitask)
         nt_val = transform_nodes(n_val_orig, node_scaler, is_multitask)
         nt_test = transform_nodes(n_test_orig, node_scaler, is_multitask)

    if graph_scaler and node_scaler:
        scaler = DualScaler(graph_scaler, node_scaler)
    elif graph_scaler:
        scaler = graph_scaler
    elif node_scaler:
        scaler = node_scaler

    if scaler_path and scaler:
        joblib.dump(scaler, scaler_path)

    # Create datasets
    train_dataset = MoleculeDataset(smiles_train, node_targets=nt_train, graph_targets=gt_train, positional_encoding_kwargs=positional_encoding_kwargs)
    val_dataset = MoleculeDataset(smiles_val, node_targets=nt_val, graph_targets=gt_val, positional_encoding_kwargs=positional_encoding_kwargs)
    test_dataset = MoleculeDataset(smiles_test, node_targets=nt_test, graph_targets=gt_test, positional_encoding_kwargs=positional_encoding_kwargs)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, val_loader, test_loader, scaler
