import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, SGConv, global_mean_pool, global_max_pool, global_add_pool, SAGPooling
from torch_geometric.nn import BatchNorm, LayerNorm
from torch_geometric.utils import degree


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, output_channels, encoder_type='GCN', pool_type='mean_pool'):
        super(GNNEncoder, self).__init__()
        self.encoder_type = encoder_type

        if encoder_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
        elif encoder_type == 'GraphSAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        elif encoder_type == 'SGC':
            self.conv1 = SGConv(in_channels, hidden_channels)
            self.conv2 = SGConv(hidden_channels, hidden_channels)
        elif encoder_type == 'GIN':
            self.mlp1 = nn.Linear(in_channels, hidden_channels)
            self.conv1 = GINConv(self.mlp1)

            self.mlp2 = nn.Linear(hidden_channels, hidden_channels)
            self.conv2 = GINConv(self.mlp2)
        else:  # GCN (default)
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)

        
        self.lins = nn.Linear(hidden_channels, output_channels)
        self.pool = Pool(hidden_channels, type=pool_type)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lins.reset_parameters()
        if self.encoder_type == 'GIN':
            self.mlp1.reset_parameters()
            self.mlp2.reset_parameters()
        
        if hasattr(self.pool, 'reset_parameters'):
            self.pool.reset_parameters()

    
    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
       
        out = self.lins(x)
        pooled = self.pool(x, edge_index, batch)
        return out, pooled

    def get_params(self):
        return OrderedDict((name, param) for name, param in self.named_parameters())

class Pool(nn.Module):
    def __init__(self, in_channels, type='mean_pool', ratio=0.5):
        super(Pool, self).__init__()
        self.type = type
        self.sag_pool = SAGPooling(in_channels, ratio)
        
    def reset_parameters(self):
        self.sag_pool.reset_parameters()
    
    def forward(self, x, edge_index, batch=None):
        if self.type == 'mean_pool':
            return global_mean_pool(x, batch)
        elif self.type == 'max_pool':
            return global_max_pool(x, batch)
        elif self.type == 'sum_pool':
            return global_add_pool(x, batch)
        elif self.type == 'sag_pool':
            x, _, _, _, _, _ = self.sag_pool(x, edge_index, batch)
            return global_mean_pool(x, batch)

class HighOrderGNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, output_channels, encoder_type='GCN', pool_type='mean_pool', fusion_type = 'attention'):
        super(HighOrderGNNEncoder, self).__init__()
        
        self.fusion_type = fusion_type
        
        if encoder_type=='GAT':
            self.conv1 = GATConv(in_channels, hidden_channels)
            self.conv2 = GATConv(in_channels, hidden_channels)
        elif encoder_type=='GraphSAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(in_channels, hidden_channels)
        elif encoder_type=='SGC':
            self.conv1 = SGConv(in_channels, hidden_channels)
            self.conv2 = SGConv(in_channels, hidden_channels)
        elif encoder_type=='GIN':
            self.mlp1 = nn.Linear(in_channels, hidden_channels)
            self.mlp2 = nn.Linear(in_channels, hidden_channels)
            self.conv1 = GINConv(self.mlp1)
            self.conv2 = GINConv(self.mlp2)
        else:#GCN
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(in_channels, hidden_channels)

    
        # Fusion mechanisms
        if fusion_type == 'attention':
            self.fusion = AttentionFusion(hidden_channels)
            final_dim = hidden_channels
        elif fusion_type == 'cross_attention':
            self.fusion = CrossViewAttention(hidden_channels)
            final_dim = 2 * hidden_channels
        elif fusion_type == 'gated':
            self.gate = nn.Linear(2 * hidden_channels, hidden_channels)
            final_dim = hidden_channels
        else:  # concat
            final_dim = 2 * hidden_channels
            
        self.pool = Pool(hidden_channels, type=pool_type)

        # Output layers with residual connections
        self.output_layer = nn.Linear(final_dim, output_channels)

        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.lins.reset_parameters()

    def forward(self, x_ego, ego_edge_index, x_cut, cut_edge_index, batch=None):
        # Apply GNN layers
        x_ego = self.conv1(x_ego, ego_edge_index)
        x_cut = self.conv2(x_cut, cut_edge_index)
       
        x_ego_pooled = self.pool(x_ego, ego_edge_index, batch)
        x_cut_pooled = self.pool(x_cut, cut_edge_index, batch)

        # Fusion
        if self.fusion_type == 'attention':
            fused_features = self.fusion(x_ego_pooled, x_cut_pooled)
        elif self.fusion_type == 'cross_attention':
            ego_attended, cut_attended = self.fusion(x_ego_pooled, x_cut_pooled)
            fused_features = torch.cat([ego_attended, cut_attended], dim=-1)
        elif self.fusion_type == 'gated':
            combined = torch.cat([x_ego_pooled, x_cut_pooled], dim=-1)
            gate_weights = torch.sigmoid(self.gate(combined))
            fused_features = gate_weights * x_ego_pooled + (1 - gate_weights) * x_cut_pooled
        else:  # concat
            fused_features = torch.cat([x_ego_pooled, x_cut_pooled], dim=-1)
        

        return  self.output_layer(fused_features)
    
    def get_params(self):
        return OrderedDict((name, param) for name, param in self.named_parameters())

class AttentionFusion(nn.Module):
    """Attention-based fusion of multiple subgraph representations."""
    def __init__(self, hidden_channels):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_channels, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_channels)
        
    def forward(self, ego_features, cut_features):
        # Stack features for attention
        features = torch.stack([ego_features, cut_features], dim=1)  # [batch, 2, hidden]
        
        # Self-attention across the two views
        attended, _ = self.attention(features, features, features)
        attended = self.norm(attended + features)  # Residual connection
        
        # Return weighted combination
        return attended.mean(dim=1)  # Average across views

class CrossViewAttention(nn.Module):
    """Cross-attention between ego and cut subgraph features."""
    def __init__(self, hidden_channels):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(hidden_channels, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        
    def forward(self, ego_features, cut_features):
        # Cross-attention: ego attends to cut
        ego_attended, _ = self.cross_attention(
            ego_features.unsqueeze(1), cut_features.unsqueeze(1), cut_features.unsqueeze(1)
        )
        ego_attended = self.norm1(ego_attended.squeeze(1) + ego_features)
        
        # Cross-attention: cut attends to ego  
        cut_attended, _ = self.cross_attention(
            cut_features.unsqueeze(1), ego_features.unsqueeze(1), ego_features.unsqueeze(1)
        )
        cut_attended = self.norm2(cut_attended.squeeze(1) + cut_features)
        
        return ego_attended, cut_attended

