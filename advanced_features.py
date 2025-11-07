# File: advanced_features.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import numpy as np

# ========================
# 1. LSTM ENCODER FOR TEMPORAL FEATURES
# ========================

class LSTMFeatureEncoder(nn.Module):
    """
    Encode temporal sequences using LSTM
    Input: [batch, seq_len, feature_dim]
    Output: [batch, hidden_dim]
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super(LSTMFeatureEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, feature_dim]
        Returns:
            output: [batch, hidden_dim]
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        return h_n[-1]  # [batch, hidden_dim]

# ========================
# 2. MULTI-HEAD ATTENTION MODULE
# ========================

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for node features
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, num_nodes, embed_dim]
        Returns:
            attended_x: [batch, num_nodes, embed_dim]
            attention_weights: [batch, num_nodes, num_nodes]
        """
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        return attn_output, attn_weights

# ========================
# 3. ADVANCED GNN WITH LSTM AND ATTENTION
# ========================

class AdvancedTemporalGNN(nn.Module):
    """
    Advanced GNN combining:
    - LSTM for temporal feature encoding
    - Multi-head attention for node interactions
    - GNN for spatial propagation
    """
    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int,
        lstm_hidden: int = 64,
        gnn_hidden: int = 64,
        num_heads: int = 4,
        num_classes: int = 2
    ):
        super(AdvancedTemporalGNN, self).__init__()
        
        # LSTM encoder for dynamic features
        self.lstm_encoder = LSTMFeatureEncoder(
            input_dim=dynamic_dim,
            hidden_dim=lstm_hidden,
            num_layers=2
        )
        
        # Combine static + LSTM-encoded dynamic features
        combined_dim = static_dim + lstm_hidden
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            embed_dim=combined_dim,
            num_heads=num_heads
        )
        
        # GNN layers
        self.conv1 = GATConv(combined_dim, gnn_hidden, heads=4, dropout=0.3)
        self.conv2 = GATConv(gnn_hidden * 4, gnn_hidden, heads=4, dropout=0.3)
        self.conv3 = GATConv(gnn_hidden * 4, gnn_hidden, heads=1, concat=False)
        
        # Output layer
        self.fc = nn.Linear(gnn_hidden, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, data, dynamic_sequence):
        """
        Args:
            data: PyG Data object with static features
            dynamic_sequence: [batch_size, num_nodes, seq_len, dynamic_dim]
        
        Returns:
            logits: [batch_size * num_nodes, num_classes]
            attention_weights: attention weights from attention layer
        """
        batch_size, num_nodes, seq_len, dynamic_dim = dynamic_sequence.shape
        
        # Encode temporal sequences with LSTM
        # Reshape: [batch_size * num_nodes, seq_len, dynamic_dim]
        dynamic_flat = dynamic_sequence.reshape(batch_size * num_nodes, seq_len, dynamic_dim)
        lstm_features = self.lstm_encoder(dynamic_flat)  # [batch_size * num_nodes, lstm_hidden]
        
        # Reshape back: [batch_size * num_nodes, lstm_hidden]
        # Combine with static features from data.x
        static_features = data.x  # [batch_size * num_nodes, static_dim]
        combined_features = torch.cat([static_features, lstm_features], dim=1)
        
        # Apply attention
        # Reshape for attention: [batch_size, num_nodes, combined_dim]
        x_attn = combined_features.reshape(batch_size, num_nodes, -1)
        x_attended, attn_weights = self.attention(x_attn)
        
        # Flatten back: [batch_size * num_nodes, combined_dim]
        x = x_attended.reshape(batch_size * num_nodes, -1)
        
        # GNN layers
        edge_index = data.edge_index
        
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        x = self.conv3(x, edge_index)
        
        # Output
        logits = self.fc(x)
        
        return logits, attn_weights

# ========================
# 4. FEATURE EXTRACTION UTILS
# ========================

def create_temporal_windows(
    dynamic_features: np.ndarray,
    window_size: int = 3
) -> np.ndarray:
    """
    Create sliding windows of temporal features
    
    Args:
        dynamic_features: [n_samples, num_nodes, max_steps, dynamic_dim]
        window_size: size of temporal window
    
    Returns:
        windowed_features: [n_samples, num_nodes, (max_steps - window_size + 1), window_size, dynamic_dim]
    """
    n_samples, num_nodes, max_steps, dynamic_dim = dynamic_features.shape
    
    windows = []
    for t in range(max_steps - window_size + 1):
        window = dynamic_features[:, :, t:t+window_size, :]
        windows.append(window)
    
    return np.array(windows).transpose(1, 2, 0, 3, 4)

def extract_lstm_embeddings(
    dynamic_features: np.ndarray,
    lstm_encoder: LSTMFeatureEncoder,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Extract LSTM embeddings for all samples
    
    Args:
        dynamic_features: [n_samples, num_nodes, max_steps, dynamic_dim]
        lstm_encoder: trained LSTM encoder
        device: computation device
    
    Returns:
        embeddings: [n_samples, num_nodes, lstm_hidden_dim]
    """
    lstm_encoder.eval()
    
    n_samples, num_nodes, max_steps, dynamic_dim = dynamic_features.shape
    embeddings = []
    
    with torch.no_grad():
        for i in range(n_samples):
            sample_embeddings = []
            for j in range(num_nodes):
                # [max_steps, dynamic_dim]
                temporal_seq = torch.tensor(
                    dynamic_features[i, j], 
                    dtype=torch.float32
                ).unsqueeze(0).to(device)  # [1, max_steps, dynamic_dim]
                
                embedding = lstm_encoder(temporal_seq)  # [1, lstm_hidden_dim]
                sample_embeddings.append(embedding.cpu().numpy())
            
            embeddings.append(np.array(sample_embeddings))
    
    return np.array(embeddings)

# ========================
# 5. EXAMPLE USAGE
# ========================

if __name__ == "__main__":
    # Example dimensions
    batch_size = 8
    num_nodes = 7
    seq_len = 5
    static_dim = 5
    dynamic_dim = 6
    
    # Create dummy data
    from torch_geometric.data import Data
    
    x_static = torch.randn(batch_size * num_nodes, static_dim)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    dynamic_seq = torch.randn(batch_size, num_nodes, seq_len, dynamic_dim)
    
    data = Data(x=x_static, edge_index=edge_index)
    
    # Initialize model
    model = AdvancedTemporalGNN(
        static_dim=static_dim,
        dynamic_dim=dynamic_dim,
        lstm_hidden=64,
        gnn_hidden=64,
        num_heads=3,
        num_classes=2
    )
    
    # Forward pass
    logits, attn_weights = model(data, dynamic_seq)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    print("\nAdvanced GNN model created successfully!")
