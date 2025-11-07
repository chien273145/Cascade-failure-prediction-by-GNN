# File: train_gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.data.data import DataEdgeAttr,DataTensorAttr

# Cho phép PyTorch load những class này
torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, DataLoader])

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# ========================
# 1. MODEL ARCHITECTURE
# ========================

class TemporalGNN(nn.Module):
    """
    Temporal GNN for cascade failure prediction
    Combines GCN layers with temporal features
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 64, 
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super(TemporalGNN, self).__init__()
        
        # GNN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # GCN layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GCN layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GCN layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Node classification
        x = self.fc(x)
        
        return x

class TemporalGAT(nn.Module):
    """
    Graph Attention Network variant with multi-head attention
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 64, 
        num_classes: int = 2,
        heads: int = 4,
        dropout: float = 0.3
    ):
        super(TemporalGAT, self).__init__()
        
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        x = self.conv3(x, edge_index)
        x = self.fc(x)
        
        return x

# ========================
# 2. TRAINING FUNCTIONS
# ========================

def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch)
        loss = criterion(out, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions
        preds = out.argmax(dim=1).cpu().numpy()
        labels = batch.y.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            
            # Predictions
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

# ========================
# 3. MAIN TRAINING LOOP
# ========================

def train_model(
    model_type: str = 'GCN',
    epochs: int = 50,
    lr: float = 0.001,
    hidden_dim: int = 64,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Main training function"""
    
    # Load data
    print("Loading dataloaders...")
    data_dict = torch.load('/content/pyg_datasets.pt', weights_only=False)
    train_loader = data_dict['train_data']
    val_loader = data_dict['val_data']
    test_loader = data_dict['test_data']
    
    # Get input dimension from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch.x.shape[1]
    
    print(f"\nInput dimension: {input_dim}")
    print(f"Device: {device}")
    
    # Initialize model
    if model_type == 'GCN':
        model = TemporalGNN(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=2)
    elif model_type == 'GAT':
        model = TemporalGAT(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_f1 = 0
    patience = 10
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"Training {model_type} model")
    print(f"{'='*60}\n")
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Print metrics
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        # Early stopping
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'best_{model_type}_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Load best model and test
    print(f"\n{'='*60}")
    print("Testing best model...")
    print(f"{'='*60}\n")
    
    model.load_state_dict(torch.load(f'train_gnn.pt'))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print("Test Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    
    return model, test_metrics

if __name__ == "__main__":
    # Train GCN
    model_gcn, metrics_gcn = train_model(model_type='GCN', epochs=50)
    
    # Train GAT
    # model_gat, metrics_gat = train_model(model_type='GAT', epochs=50)
