# File: app.py

from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import json
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Import your models
from best_GCN_model import TemporalGNN, TemporalGAT
from pyg_datasets import create_edge_index, create_edge_attr
from torch_geometric.data import Data

app = Flask(__name__)

# ========================
# 1. LOAD TRAINED MODEL
# ========================

class ModelInference:
    def __init__(self, model_path='best_GCN_model.pt', model_type='GCN'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model architecture (you need to know input_dim)
        self.input_dim = 11  # 5 static + 6 dynamic features
        self.model_type = model_type
        
        if model_type == 'GCN':
            self.model = TemporalGNN(input_dim=self.input_dim, hidden_dim=64, num_classes=2)
        else:
            self.model = TemporalGAT(input_dim=self.input_dim, hidden_dim=64, num_classes=2)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load graph structure
        data = np.load('thermal_power_cascade_dataset.npz', allow_pickle=True)
        self.adj_matrix = data['adjacency_matrix']
        self.node_names = data['node_names'].tolist()
        self.num_nodes = len(self.node_names)
        
    def predict(self, static_features, dynamic_features):
        """
        Predict failure probability for each node
        
        Args:
            static_features: [num_nodes, static_dim]
            dynamic_features: [num_nodes, dynamic_dim]
        
        Returns:
            predictions: [num_nodes] - probability of failure
        """
        # Combine features
        node_features = np.concatenate([static_features, dynamic_features], axis=1)
        x = torch.tensor(node_features, dtype=torch.float).to(self.device)
        
        # Create graph
        edge_index = create_edge_index(self.adj_matrix)
        edge_attr = create_edge_attr(self.adj_matrix, edge_index)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data = data.to(self.device)
        
        # Predict
        with torch.no_grad():
            out = self.model(data)
            probs = torch.softmax(out, dim=1)[:, 1]  # Probability of failure
        
        return probs.cpu().numpy()
    
    def simulate_cascade(self, init_failure_idx, num_steps=5):
        """
        Simulate cascade propagation for visualization
        
        Args:
            init_failure_idx: initial failure node
            num_steps: number of time steps to simulate
        
        Returns:
            cascade_sequence: [num_nodes, num_steps] - failure states over time
            probabilities: [num_nodes, num_steps] - failure probabilities
        """
        cascade_sequence = np.zeros((self.num_nodes, num_steps))
        probabilities = np.zeros((self.num_nodes, num_steps))
        
        # Initialize
        cascade_sequence[init_failure_idx, 0] = 1
        
        # Load nominal features (simplified - in production, use real sensor data)
        static_features = np.random.rand(self.num_nodes, 5)
        
        for t in range(num_steps):
            # Create current state features
            dynamic_features = np.random.randn(self.num_nodes, 6)  # Simplified
            
            # Predict
            probs = self.predict(static_features, dynamic_features)
            probabilities[:, t] = probs
            
            # Update cascade (threshold-based)
            if t < num_steps - 1:
                new_failures = (probs > 0.5) & (cascade_sequence[:, t] == 0)
                cascade_sequence[:, t+1] = cascade_sequence[:, t].copy()
                cascade_sequence[new_failures, t+1] = 1
        
        return cascade_sequence, probabilities

# Initialize model
model_inference = ModelInference(model_path='best_GCN_model.pt', model_type='GCN')

# ========================
# 2. VISUALIZATION FUNCTIONS
# ========================

def create_cascade_plot(cascade_sequence, node_names):
    """Create cascade heatmap and return as base64 image"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(
        cascade_sequence,
        cmap='RdYlGn_r',
        yticklabels=node_names,
        xticklabels=[f't={i}' for i in range(cascade_sequence.shape[1])],
        cbar_kws={'label': 'Failure State'},
        ax=ax
    )
    
    ax.set_title('Cascade Failure Propagation', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Equipment')
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return image_base64

def create_network_plot(adj_matrix, node_names, failure_states, probabilities):
    """Create network graph visualization"""
    G = nx.Graph()
    
    for i, name in enumerate(node_names):
        G.add_node(i, label=name, failed=failure_states[i], prob=probabilities[i])
    
    for i in range(len(node_names)):
        for j in range(i+1, len(node_names)):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j)
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Node colors based on probability
    node_colors = ['red' if fs == 1 else f'#{int(255*(1-p)):02x}{int(255*p):02x}00' 
                   for fs, p in zip(failure_states, probabilities)]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500, ax=ax)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, 
                           labels={i: name for i, name in enumerate(node_names)},
                           font_size=9, ax=ax)
    
    ax.set_title('Power Plant Network - Failure Risk', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return image_base64

# ========================
# 3. API ROUTES
# ========================

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', node_names=model_inference.node_names)

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for prediction
    Expects JSON: {
        "init_failure": node_index,
        "num_steps": 5
    }
    """
    try:
        data = request.get_json()
        init_failure = int(data.get('init_failure', 0))
        num_steps = int(data.get('num_steps', 5))
        
        # Simulate cascade
        cascade_seq, probs = model_inference.simulate_cascade(init_failure, num_steps)
        
        # Create visualizations
        cascade_plot = create_cascade_plot(cascade_seq, model_inference.node_names)
        network_plot = create_network_plot(
            model_inference.adj_matrix,
            model_inference.node_names,
            cascade_seq[:, -1],  # Final state
            probs[:, -1]  # Final probabilities
        )
        
        # Prepare response
        response = {
            'success': True,
            'cascade_sequence': cascade_seq.tolist(),
            'probabilities': probs.tolist(),
            'cascade_plot': cascade_plot,
            'network_plot': network_plot,
            'node_names': model_inference.node_names,
            'summary': {
                'total_failures': int(cascade_seq[:, -1].sum()),
                'high_risk_nodes': [
                    model_inference.node_names[i] 
                    for i in range(len(probs[:, -1])) 
                    if probs[i, -1] > 0.7
                ]
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/equipment-status', methods=['POST'])
def equipment_status():
    """
    Get current equipment status and predictions
    Expects JSON with sensor readings
    """
    try:
        data = request.get_json()
        
        # Extract features from request
        # In production, this would come from real sensor data
        static_features = np.array(data.get('static_features', np.random.rand(7, 5)))
        dynamic_features = np.array(data.get('dynamic_features', np.random.randn(7, 6)))
        
        # Predict
        probs = model_inference.predict(static_features, dynamic_features)
        
        # Create response
        equipment_status = []
        for i, name in enumerate(model_inference.node_names):
            equipment_status.append({
                'name': name,
                'risk_level': 'High' if probs[i] > 0.7 else 'Medium' if probs[i] > 0.4 else 'Low',
                'probability': float(probs[i]),
                'status': 'Warning' if probs[i] > 0.5 else 'Normal'
            })
        
        return jsonify({
            'success': True,
            'equipment_status': equipment_status
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
