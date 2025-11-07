# File: convert_to_pyg.py

import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from typing import List, Tuple

def load_dataset(file_path: str = 'thermal_power_cascade_dataset.npz'):
    """Load the generated dataset"""
    data = np.load(file_path, allow_pickle=True)
    return {
        'static_features': data['static_features'],
        'dynamic_features': data['dynamic_features'],
        'labels': data['labels'],
        'init_failures': data['init_failures'],
        'adjacency_matrix': data['adjacency_matrix'],
        'node_names': data['node_names']
    }

def create_edge_index(adjacency_matrix: np.ndarray) -> torch.Tensor:
    """
    Convert adjacency matrix to edge_index format for PyG
    Returns: [2, num_edges] tensor
    """
    edges = np.where(adjacency_matrix > 0)
    edge_index = torch.tensor(np.array(edges), dtype=torch.long)
    return edge_index

def create_edge_attr(adjacency_matrix: np.ndarray, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Create edge attributes (edge types: critical=1, supportive=2)
    Returns: [num_edges, 1] tensor
    """
    edge_attr = []
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        edge_type = adjacency_matrix[src, dst]
        edge_attr.append([edge_type])
    return torch.tensor(edge_attr, dtype=torch.float)

def convert_sample_to_pyg(
    static_features: np.ndarray,
    dynamic_features: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: np.ndarray,
    init_failure: int,
    time_step: int
) -> Data:
    """
    Convert one sample at one time step to PyG Data object

    Args:
        static_features: [num_nodes, static_dim]
        dynamic_features: [num_nodes, max_steps, dynamic_dim]
        labels: [num_nodes, max_steps]
        adjacency_matrix: [num_nodes, num_nodes]
        init_failure: int (initial failure node)
        time_step: int (which time step to use)

    Returns:
        PyG Data object
    """
    num_nodes = static_features.shape[0]

    # Node features: concatenate static + dynamic at current time step
    dynamic_t = dynamic_features[:, time_step, :]  # [num_nodes, dynamic_dim]
    node_features = np.concatenate([static_features, dynamic_t], axis=1)
    x = torch.tensor(node_features, dtype=torch.float)

    # Labels at current time step
    y = torch.tensor(labels[:, time_step], dtype=torch.long)

    # Edge index and attributes
    edge_index = create_edge_index(adjacency_matrix)
    edge_attr = create_edge_attr(adjacency_matrix, edge_index)

    # Additional info
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        init_failure=torch.tensor([init_failure], dtype=torch.long),
        num_nodes=num_nodes
    )

    return data

def create_temporal_graphs(
    static_features: np.ndarray,
    dynamic_features: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: np.ndarray,
    init_failures: np.ndarray,
    prediction_horizon: int = 1
) -> List[Data]:
    """
    Create sequence of temporal graphs for all samples
    Each sample creates (max_steps - prediction_horizon) graphs

    Args:
        prediction_horizon: predict labels at t+h given features at t

    Returns:
        List of PyG Data objects
    """
    n_samples, num_nodes, max_steps, _ = dynamic_features.shape
    graph_list = []

    for sample_idx in range(n_samples):
        for t in range(max_steps - prediction_horizon):
            # Use features at time t to predict labels at time t + prediction_horizon
            data = convert_sample_to_pyg(
                static_features=static_features,
                dynamic_features=dynamic_features[sample_idx],
                labels=labels[sample_idx],
                adjacency_matrix=adjacency_matrix,
                init_failure=init_failures[sample_idx],
                time_step=t
            )

            # Update label to be prediction target
            data.y = torch.tensor(
                labels[sample_idx, :, t + prediction_horizon],
                dtype=torch.long
            )

            graph_list.append(data)

        if (sample_idx + 1) % 100 == 0:
            print(f"Processed {sample_idx + 1}/{n_samples} samples")

    return graph_list

def split_dataset(graph_list: List[Data], train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Split dataset into train/val/test"""
    n = len(graph_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_data = graph_list[:n_train]
    val_data = graph_list[n_train:n_train + n_val]
    test_data = graph_list[n_train + n_val:]

    return train_data, val_data, test_data

if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset('thermal_power_cascade_dataset.npz')

    print("Converting to PyTorch Geometric format...")
    graph_list = create_temporal_graphs(
        static_features=dataset['static_features'],
        dynamic_features=dataset['dynamic_features'],
        labels=dataset['labels'],
        adjacency_matrix=dataset['adjacency_matrix'],
        init_failures=dataset['init_failures'],
        prediction_horizon=1
    )

    print(f"\nTotal graphs created: {len(graph_list)}")
    print(f"Sample graph: {graph_list[0]}")

    # Split dataset
    train_data, val_data, test_data = split_dataset(graph_list)
    print(f"\nTrain: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create DataLoaders (optional, can be created when needed)
    # train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    # test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Save the dataset lists instead of DataLoaders
    torch.save({
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data
    }, 'convert_to_pyg.pt')

    print("\nDatasets saved to 'convert_to_pyg.py'")
