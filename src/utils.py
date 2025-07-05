import torch
import torch.nn.functional as F

import numpy as np

from torch_geometric.utils import k_hop_subgraph, subgraph

from sklearn.model_selection import train_test_split

def create_train_valid_test_split(data, seed, train_size, val_size, test_size, n_way):
    torch.manual_seed(seed)
    np.random.seed(seed)

    unique_classes = torch.unique(data.y).tolist()
    np.random.shuffle(unique_classes)

    total_available = len(unique_classes)
    assert train_size + test_size <= total_available, "Too many classes requested for train + test."

    train_classes = unique_classes[:train_size]
    remaining_classes = unique_classes[train_size:]

    # Use remaining classes for validation and test
    max_val_test_size = total_available - train_size

    if (val_size + test_size) <= max_val_test_size:
        # Enough for clean split
        valid_classes = remaining_classes[:val_size]
        test_classes = remaining_classes[val_size:val_size + test_size]
    else:
        # Not enough unique classes — allow overlap
        valid_classes = remaining_classes[:val_size]

        # Compute test range (may overlap)
        test_start = max(0, val_size - ((val_size + test_size) - max_val_test_size))
        test_end = test_start + test_size
        test_classes = remaining_classes[test_start:test_end]

    # Final sanity check
    assert len(train_classes) >= n_way
    assert len(valid_classes) >= n_way, f"Only {len(valid_classes)} val classes, need at least {n_way}"
    assert len(test_classes) >= n_way, f"Only {len(test_classes)} test classes, need at least {n_way}"

    return train_classes, valid_classes, test_classes


def create_train_test_split(data, seed, train_size, test_size):
    # Get unique classes
    unique_classes = torch.unique(data.y)
    unique_classes = unique_classes.cpu().numpy()
    np.random.shuffle(unique_classes)
    train_classes, test_classes = train_test_split(
        unique_classes,
        train_size=train_size,
        test_size=test_size,
        random_state=seed
    )
    return train_classes, test_classes

def create_tasks(data, classes, n_way, k_shot, n_query, n_tasks):
    tasks = []
    
    n_way = min(len(classes), n_way)
    for _ in range(n_tasks):
        # Randomly select n_way classes
        selected_classes = np.random.choice(classes, n_way, replace=False)
        support_examples = []
        query_examples = []
        
        for cls in selected_classes:
            # Get indices for current class
            cls_indices = torch.where(data.y.cpu() == cls)[0].numpy()
            
            # Randomly select k_shot + n_query examples
            selected_indices = np.random.choice(
                cls_indices,
                k_shot + n_query,
                replace=False
            )
            
            support_examples.extend(selected_indices[:k_shot])
            query_examples.extend(selected_indices[k_shot:k_shot + n_query])
        
        tasks.append({
            'support': torch.tensor(support_examples, device=data.x.device),
            'query': torch.tensor(query_examples, device=data.x.device)
        })
    
    return tasks

def contrastive_loss(anchor_emb, positive_emb, negative_embs, temperature=0.1):
    
    criterion = torch.nn.CrossEntropyLoss()

    anchor_emb = F.normalize(anchor_emb, dim=-1)
    positive_emb = F.normalize(positive_emb, dim=-1)
    negative_embs = F.normalize(negative_embs, dim=-1)

    pos_sim = (anchor_emb * positive_emb).sum(dim=1) / temperature
    neg_sim = (anchor_emb * negative_embs).sum(dim=1) / temperature
    
    logits = torch.stack([pos_sim, neg_sim], dim=1)  # [B, 2]
    labels = torch.zeros(anchor_emb.size(0), dtype=torch.long, device=anchor_emb.device)
    return criterion(logits, labels) 



def get_ego_subgraph(node_idx, data, num_hops=2, max_nodes=None):
    """
    Extract ego subgraph with optional node limit for efficiency.
    """
    x, edge_index = data.x, data.edge_index
    node_indexes = torch.tensor([node_idx], device=edge_index.device)
    
    node_ids, sub_edge_index, _, _ = k_hop_subgraph(
        node_indexes,
        num_hops,
        edge_index,
        relabel_nodes=True,
        num_nodes=x.size(0)
    )
    
    # Optional: Limit subgraph size for computational efficiency
    if max_nodes is not None and len(node_ids) > max_nodes:
        # Keep target node and randomly sample others
        target_mask = node_ids == node_idx
        other_nodes = node_ids[~target_mask]
        keep_count = max_nodes - 1  # -1 for target node
        
        if len(other_nodes) > keep_count:
            keep_indices = torch.randperm(len(other_nodes))[:keep_count]
            other_nodes = other_nodes[keep_indices]
        
        node_ids = torch.cat([node_ids[target_mask], other_nodes])
        sub_edge_index, _ = subgraph(node_ids, edge_index, relabel_nodes=True)

    target_subgraph_idx = (node_ids == node_idx).nonzero(as_tuple=True)[0].item()
    return target_subgraph_idx, x[node_ids], sub_edge_index

def get_structural_subgraph(node_idx, data, num_hops=2, strategy='high_degree'):
    """
    Extract subgraph based on structural properties rather than random edge removal.
    
    Strategies:
    - 'high_degree': Keep edges connected to high-degree nodes
    - 'low_degree': Keep edges connected to low-degree nodes  
    - 'high_centrality': Keep edges based on betweenness centrality
    - 'random_walk': Use random walk to select important edges
    """
    x = data.x
    edge_index = data.edge_index
    
    # Calculate node degrees
    degrees = torch.bincount(edge_index.flatten(), minlength=x.size(0))
    
    if strategy == 'high_degree':
        # Keep edges connected to high-degree nodes
        edge_degrees = degrees[edge_index[0]] + degrees[edge_index[1]]
        k = edge_index.size(1) // 2
        keep_indices = torch.topk(edge_degrees, k=k, largest=True).indices
        
    elif strategy == 'low_degree':
        # Keep edges connected to low-degree nodes
        edge_degrees = degrees[edge_index[0]] + degrees[edge_index[1]]
        k = edge_index.size(1) // 2
        keep_indices = torch.topk(edge_degrees, k=k, largest=False).indices
        
    elif strategy == 'centrality_based':
        # Use edge betweenness (approximated by degree product)
        edge_centrality = degrees[edge_index[0]] * degrees[edge_index[1]]
        k = edge_index.size(1) // 2
        keep_indices = torch.topk(edge_centrality, k=k, largest=True).indices
        
    elif strategy == 'distance_based':
        # Prefer edges at different hop distances from target
        node_indexes = torch.tensor([node_idx], device=edge_index.device)
        
        # Get 1-hop and 2-hop neighbors
        one_hop_nodes, _, _, _ = k_hop_subgraph(node_indexes, 1, edge_index, num_nodes=x.size(0))
        two_hop_nodes, _, _, _ = k_hop_subgraph(node_indexes, 2, edge_index, num_nodes=x.size(0))
        
        # Create edge weights based on distance from target
        edge_weights = torch.zeros(edge_index.size(1), device=edge_index.device)
        
        for i, (src, dst) in enumerate(edge_index.t()):
            src_in_1hop = src in one_hop_nodes
            dst_in_1hop = dst in one_hop_nodes
            src_in_2hop = src in two_hop_nodes
            dst_in_2hop = dst in two_hop_nodes
            
            # Higher weight for edges spanning different hop distances
            if (src_in_1hop and dst_in_2hop) or (src_in_2hop and dst_in_1hop):
                edge_weights[i] = 2.0
            elif src_in_1hop and dst_in_1hop:
                edge_weights[i] = 1.0
            else:
                edge_weights[i] = 0.5
                
        k = edge_index.size(1) // 2
        keep_indices = torch.topk(edge_weights, k=k, largest=True).indices
        
    else:  # random (fallback)
        keep_indices = torch.randperm(edge_index.size(1))[:edge_index.size(1) // 2]

    new_edge_index = edge_index[:, keep_indices]
    
    # Extract subgraph around target node
    node_indexes = torch.tensor([node_idx], device=edge_index.device)
    node_ids, sub_edge_index, _, _ = k_hop_subgraph(
        node_indexes,
        num_hops,
        new_edge_index,
        relabel_nodes=True,
        num_nodes=x.size(0)
    )

    target_subgraph_idx = (node_ids == node_idx).nonzero(as_tuple=True)[0].item()
    return target_subgraph_idx, x[node_ids], sub_edge_index

def get_complementary_subgraphs(node_idx, data, num_hops=2, complement_strategy='spectral'):
    """
    Extract two complementary subgraphs that capture different aspects of the graph structure.
    
    Strategies:
    - 'spectral': Use spectral clustering to split edges
    - 'community': Use community detection
    - 'local_global': One local (1-hop), one global (2-hop+)
    """
    x = data.x
    edge_index = data.edge_index
    
    if complement_strategy == 'local_global':
        # First subgraph: local (1-hop)
        node_indexes = torch.tensor([node_idx], device=edge_index.device)
        local_nodes, local_edges, _, _ = k_hop_subgraph(
            node_indexes, 1, edge_index, relabel_nodes=True, num_nodes=x.size(0)
        )
        
        # Second subgraph: global (2-hop but skip direct neighbors)
        global_nodes, global_edges_temp, _, _ = k_hop_subgraph(
            node_indexes, num_hops, edge_index, relabel_nodes=False, num_nodes=x.size(0)
        )
        
        # Remove 1-hop edges from global subgraph
        one_hop_set = set(local_nodes.tolist())
        global_edge_mask = ~((torch.isin(edge_index[0], local_nodes)) & 
                            (torch.isin(edge_index[1], local_nodes)))
        global_edge_index_filtered = edge_index[:, global_edge_mask]
        
        global_nodes, global_edges, _, _ = k_hop_subgraph(
            node_indexes, num_hops, global_edge_index_filtered, 
            relabel_nodes=True, num_nodes=x.size(0)
        )
        
        target_local_idx = (local_nodes == node_idx).nonzero(as_tuple=True)[0].item()
        target_global_idx = (global_nodes == node_idx).nonzero(as_tuple=True)[0].item()
        
        return (target_local_idx, x[local_nodes], local_edges), \
               (target_global_idx, x[global_nodes], global_edges)
               
    elif complement_strategy == 'degree_split':
        # Split based on node degrees
        degrees = torch.bincount(edge_index.flatten(), minlength=x.size(0))
        median_degree = degrees.float().median()
        
        # High-degree subgraph
        high_degree_mask = degrees >= median_degree
        high_degree_nodes = torch.nonzero(high_degree_mask).flatten()
        high_degree_edge_mask = (torch.isin(edge_index[0], high_degree_nodes) | 
                                torch.isin(edge_index[1], high_degree_nodes))
        high_degree_edges = edge_index[:, high_degree_edge_mask]
        
        # Low-degree subgraph  
        low_degree_edges = edge_index[:, ~high_degree_edge_mask]
        
        # Extract k-hop subgraphs from each
        node_indexes = torch.tensor([node_idx], device=edge_index.device)
        
        high_nodes, high_sub_edges, _, _ = k_hop_subgraph(
            node_indexes, num_hops, high_degree_edges, relabel_nodes=True, num_nodes=x.size(0)
        )
        low_nodes, low_sub_edges, _, _ = k_hop_subgraph(
            node_indexes, num_hops, low_degree_edges, relabel_nodes=True, num_nodes=x.size(0)
        )
        
        target_high_idx = (high_nodes == node_idx).nonzero(as_tuple=True)[0].item()
        target_low_idx = (low_nodes == node_idx).nonzero(as_tuple=True)[0].item()
        
        return (target_high_idx, x[high_nodes], high_sub_edges), \
               (target_low_idx, x[low_nodes], low_sub_edges)
    
    else:  # Default to original approach but improved
        return get_ego_subgraph(node_idx, data, num_hops), \
               get_structural_subgraph(node_idx, data, num_hops, 'high_degree')

def adaptive_subgraph_extraction(node_idx, data, num_hops=2, node_budget=100):
    """
    Adaptive subgraph extraction that adjusts based on local graph density.
    """
    x = data.x
    edge_index = data.edge_index
    
    # Get initial ego subgraph to assess local density
    node_indexes = torch.tensor([node_idx], device=edge_index.device)
    initial_nodes, initial_edges, _, _ = k_hop_subgraph(
        node_indexes, 1, edge_index, relabel_nodes=False, num_nodes=x.size(0)
    )
    
    # Calculate local density
    local_density = initial_edges.size(1) / (len(initial_nodes) * (len(initial_nodes) - 1) + 1e-8)
    
    # Adjust strategy based on density
    if local_density > 0.3:  # High density - use more selective extraction
        return get_complementary_subgraphs(node_idx, data, num_hops, 'degree_split')
    elif local_density < 0.1:  # Low density - expand search
        return get_ego_subgraph(node_idx, data, num_hops + 1), \
               get_structural_subgraph(node_idx, data, num_hops + 1, 'centrality_based')
    else:  # Medium density - standard approach
        return get_ego_subgraph(node_idx, data, num_hops), \
               get_structural_subgraph(node_idx, data, num_hops, 'high_degree')
