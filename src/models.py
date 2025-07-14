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
