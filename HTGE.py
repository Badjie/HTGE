import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

class MGDSGU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=1):
        super(MGDSGU, self).__init__()
        self.hidden_dim = hidden_dim
        
        # SAGEConv layers for transforming input and hidden states
        self.sage_input = SAGEConv(input_dim, hidden_dim)#, heads=num_heads)
        self.sage_hidden = SAGEConv(hidden_dim, hidden_dim)#, heads=1)
        
        # single input gate (x_t) components
        self.sage_single_x = SAGEConv(input_dim, hidden_dim)#, heads=1)
        self.sage_single_h = SAGEConv(hidden_dim, hidden_dim)#, heads=1)
        
        # Hidden Update gate components
        self.sage_update_x = SAGEConv(input_dim, hidden_dim)#, heads=1)
        self.sage_update_h = SAGEConv(hidden_dim, hidden_dim)#, heads=1)
        
        # Candidate_hidden state components
        self.sage_candidate_x = SAGEConv(input_dim, hidden_dim)#, heads=1)
        self.sage_candidate_h = SAGEConv(hidden_dim, hidden_dim)#, heads=1)
        
        #linear transdormations for dimension adjustment
        self.linear_input = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.Linear(input_dim, hidden_dim)
        
        # Missing information prediction components
        self.W_gamma1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_gamma2 = nn.Linear(hidden_dim, hidden_dim)
        self.W_beta1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_beta2 = nn.Linear(hidden_dim, hidden_dim)
        
    def predict_missing_info(self, h_v, h_N_v):
        # Calculate scaling factor gamma
        gamma = torch.tanh(self.W_gamma1(h_v) + self.W_gamma2(h_N_v))
        
        # Calculate shift factor beta
        beta = torch.tanh(self.W_beta1(h_v) + self.W_beta2(h_N_v))
        
        # Global transfer information (to be learned during training)
        r = torch.zeros_like(h_v)
        
        # Calculate transfer information
        r_v = (gamma + 1) * r + beta
        
        # Calculate missing information
        m_v = h_v + r_v - h_N_v
        
        return m_v
        
    def forward(self, x, edge_index, h, batch=None):
        # Apply SAGEConv to input
        x_transformed = self.sage_input(x, edge_index)
        #x_transformed = self.linear_input(x)
        
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        
        # Calculate neighborhood information
        h_N = self.sage_hidden(h, edge_index)
        
        # Predict missing information
        m = self.predict_missing_info(h, h_N)
        
        # Update h_N with missing information
        h_N = h_N + m
        
        # Single gate
        r = torch.sigmoid(self.sage_single_x(x, edge_index) + 
                         self.sage_single_h(h_N, edge_index))
        
        # Hidden Sate Update gate
        z = torch.sigmoid(self.sage_update_x(x, edge_index) + 
                         self.sage_update_h(h_N, edge_index))
        
        # Candidate state
        h_tilde = torch.tanh(self.sage_candidate_x(x, edge_index) + 
                            self.sage_candidate_h(r * h_N, edge_index))
        
        # New hidden state
        h_new = (1 - z) * h_N + z * h_tilde
        
        return h_new


class EdgeClassifier(nn.Module):
    def __init__(self, node_embedding_dim):
        super(EdgeClassifier, self).__init__()
        # Edge embedding dimension will be 2 * node_embedding_dim (concatenated)
        edge_embedding_dim = 2 * node_embedding_dim
        
        # Two-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(edge_embedding_dim, edge_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(edge_embedding_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, edge_embeddings):
        return self.mlp(edge_embeddings)

class EnhancedTemporalGraphNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(EnhancedTemporalGraphNetwork, self).__init__()
        
        self.mgdsgu = MGDSGU(input_dim, hidden_dim)
        self.num_layers = num_layers
        self.edge_classifier = EdgeClassifier(hidden_dim)
        
    def create_edge_embeddings(self, node_embeddings, edge_index):
        # Get source and target node embeddings
        src_embeddings = node_embeddings[edge_index[0]]
        dst_embeddings = node_embeddings[edge_index[1]]
        
        # Concatenate source and target embeddings
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
        return edge_embeddings
    
    def forward(self, x, edge_index, batch=None):
        h = None
        
        # Get final node embeddings from MGGRU
        for _ in range(self.num_layers):
            h = self.mgdsgu(x, edge_index, h, batch)
        
        # Create edge embeddings by concatenating node embeddings
        edge_embeddings = self.create_edge_embeddings(h, edge_index)
        
        # Get classification probabilities
        edge_predictions = self.edge_classifier(edge_embeddings)
        
        return edge_predictions, h

def create_edge_labels(G, labels, edge_index, node_to_idx):
    edge_labels = []
    for i in range(edge_index.size(1)):
        src_idx = edge_index[0][i].item()
        dst_idx = edge_index[1][i].item()
        
        # Convert indices back to original node IDs
        src_node = list(G.nodes())[src_idx]
        dst_node = list(G.nodes())[dst_idx]
        
        # Create edge label based on source and destination node labels
        src_label = 1 if labels[src_node][0] == 'illegitimate_account' else 0
        dst_label = 1 if labels[dst_node][0] == 'illegitimate_account' else 0
        
        # Edge is labeled as illegitimate if either source or destination is illegitimate
        edge_labels.append(float(src_label or dst_label))
    
    return torch.tensor(edge_labels, dtype=torch.float)

def weighted_cross_entropy_loss(predictions, targets, pos_weight):
    """
    Custom weighted cross entropy loss
    N1: number of positive samples
    N2: number of negative samples
    """
    epsilon = 1e-7  # Small constant to prevent log(0)
    predictions = torch.clamp(predictions, epsilon, 1 - epsilon)
    
    # Calculate weighted loss
    loss = -(pos_weight * targets * torch.log(predictions) + 
             (1 - targets) * torch.log(1 - predictions))
    
    return loss.mean()

def create_graph(data):
    G = nx.DiGraph()
    nodes = set(data['from_address'].tolist() + data['to_address'].tolist())
    G.add_nodes_from(nodes)
    for _, row in data.iterrows():
        G.add_edge(row['from_address'], row['to_address'], weight=row['timestamp'])
    return G

def calculate_fraud_and_antifraud_scores(G):
    #Betweenness Centrality
    #  Note:  This is computationally expensive for large graphs.  It might be suitable for smaller, ego-networks.
    betweenness = nx.betweenness_centrality(G) # Nodes on many shortest paths indicate fraud

    fraud_scores = nx.out_degree_centrality(G)  # Out-degree, as before
    antifraud_scores = {node: 1-betweenness[node] for node in G.nodes()}  # Nodes *not* on many shortest paths
    return fraud_scores, antifraud_scores

def label_nodes(fraud_scores, antifraud_scores, fraud_threshold=0.01, antifraud_threshold=0.01):
    labels = {}
    for node in fraud_scores:
        account_label = 'illegitimate_account' if fraud_scores[node] > fraud_threshold else 'legitimate_account'
        pay_label = 'legitimate_payment' if antifraud_scores[node] > antifraud_threshold else 'illegitimate_payment'
        labels[node] = (account_label, pay_label)
    return labels

def label_edges(G):
    ego_networks = defaultdict(nx.DiGraph)
    for u, v, data in G.edges(data=True):
        ego_networks[u].add_edge(u, v, weight=data['weight'])
        ego_networks[v].add_edge(u, v, weight=data['weight'])
    return ego_networks

def count_edges(ego_network, label):
    count = 0
    for u, v, data in ego_network.edges(data=True):
        if label in data:
            count += 1
    return count

def common_eval(ego_networks):
    neighbors = {}
    for node, ego_net in ego_networks.items():
        neighbors[node] = list(ego_net.neighbors(node))
    return neighbors

def extract_features(G, node):
    ego_networks = label_edges(G)
    neighbors = common_eval(ego_networks)
    P1 = count_edges(ego_networks[node], label='legitimate_account')
    P2 = count_edges(ego_networks[node], label='illegitimate_account')
    P3 = count_edges(ego_networks[node], label='legitimate_payment')
    P4 = count_edges(ego_networks[node], label='illegitimate_payment')
    
    node_features = [P1, P2, len(neighbors.get(node, [])), 
                    P3, P4, len(neighbors.get(node, [])), 
                    G.in_degree(node), G.out_degree(node)]
    return node_features


def create_data_list(G_list, labels_list):
    data_list = []
    for G, labels in zip(G_list, labels_list):
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes)}
        
        # Create edge index for the entire graph
        edge_index = torch.tensor([(node_to_idx[u], node_to_idx[v]) 
                                  for u, v in G.edges], dtype=torch.long).t().contiguous()
        
        # Create feature matrix for all nodes
        x = torch.tensor([extract_features(G, node) for node in G.nodes], 
                        dtype=torch.float)
        
        # Create edge labels
        edge_labels = create_edge_labels(G, labels, edge_index, node_to_idx)
        
        # Create a single Data object for the entire graph
        data = Data(x=x, edge_index=edge_index, y=edge_labels)
        data_list.append(data)
    
    return data_list

def train_model(model, train_loader, epochs=100, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(epochs):
        for data in train_loader:
            optimizer.zero_grad()
            edge_predictions, _ = model(data.x, data.edge_index, data.batch)
            loss = weighted_cross_entropy_loss(edge_predictions, data.y, pos_weight=1.0)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model

def evaluate_model(model, loader):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            edge_predictions, _ = model(batch.x, batch.edge_index, batch.batch)
            all_predictions.append(edge_predictions.squeeze().cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    auc_score = roc_auc_score(all_labels, all_predictions)
    precision = precision_score(all_labels, [1 if p > 0.5 else 0 for p in all_predictions])
    recall = recall_score(all_labels, [1 if p > 0.5 else 0 for p in all_predictions])
    f1 = f1_score(all_labels, [1 if p > 0.5 else 0 for p in all_predictions])
    return auc_score, precision, recall, f1


def main():
    
    # Load the dataset
    file_path = 'E:/ansu/ERC20-stablecoins/soc-sign-bitcoinalpha.csv'
    #file_path = 'token_transfers_V3.0.0.csv'
    data = pd.read_csv(file_path)

    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

    # Sort data by timestamp
    data = data.sort_values(by='timestamp')

    # Split data into 31-day time slices
    data['time_slice'] = (data['timestamp'] - data['timestamp'].min()).dt.days // 31

    # Combine the last few sparse time slices into a single time slice
    threshold = 5  # Combine slices with fewer than 20 entries
    combined_time_slice = max(data['time_slice']) - 1
    data.loc[data['time_slice'] >= combined_time_slice, 'time_slice'] = combined_time_slice

    # Verify the new distribution of entries across time slices
    new_time_slice_counts = data['time_slice'].value_counts().sort_index()

    # Display the new distribution
    print(new_time_slice_counts)

    
    # Create a list of graphs and labels for each time slice
    G_list = []
    labels_list = []
    
    for time_slice in data['time_slice'].unique():
        slice_data = data[data['time_slice'] == time_slice]
        G = create_graph(slice_data)
        fraud_scores, antifraud_scores = calculate_fraud_and_antifraud_scores(G)
        labels = label_nodes(fraud_scores, antifraud_scores)
        G_list.append(G)
        labels_list.append(labels)
    
    # Check if we have enough data slices for splitting
    if len(G_list) > 2:
        # Split data into train, validation, and test sets (60%, 20%, 20%)
        train_G, temp_G, train_labels, temp_labels = train_test_split(G_list, labels_list, test_size=0.4, shuffle=False)
        val_G, test_G, val_labels, test_labels = train_test_split(temp_G, temp_labels, test_size=0.5, shuffle=False)
    else:
        # If there's not enough time slices, use all data for training and skip validation/testing
        train_G, train_labels = G_list, labels_list
        val_G, val_labels, test_G, test_labels = [], [], [], []
    
    # Create dataset and dataloader
    train_data_list = create_data_list(train_G, train_labels)
    val_data_list = create_data_list(val_G, val_labels) if val_G else []
    test_data_list = create_data_list(test_G, test_labels) if test_G else []
    
    train_loader = DataLoader(train_data_list, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=16, shuffle=False) if val_data_list else None
    test_loader = DataLoader(test_data_list, batch_size=16, shuffle=False) if test_data_list else None
    
    # Initialize and train model
    model = EnhancedTemporalGraphNetwork(
        input_dim=8,
        hidden_dim=16,
        num_layers=2
    )
    
    # Train the model
    trained_model = train_model(model, train_loader)
    
    # Evaluate the model on the test set if it exists
    if test_loader:
        test_auc, precision, recall, f1 = evaluate_model(trained_model, test_loader)
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
    else:
        print("Not enough data to create a test set.")
    
    return trained_model

if __name__ == "__main__":
    trained_model = main()

