import torch
import math
import torch.nn.functional as F

import numpy as np

from torch_geometric.utils import k_hop_subgraph, subgraph

from sklearn.model_selection import train_test_split

def create_train_valid_test_split(data, seed, n_way):
    torch.manual_seed(seed)
    np.random.seed(seed)

    unique_classes = torch.unique(data.y).tolist()
    np.random.shuffle(unique_classes)

    total_available = len(unique_classes)

    min_classes = n_way
    # Calculate remaining classes after satisfying minimums
    # Val can overlap with test
    remaining_classes = total_available - (2 * min_classes)
    print(remaining_classes)

    if remaining_classes > 0:
        # Distribute remaining classes according to ratios (60%, 30%, 30%)
        # Note: val and test get equal priority, so we split remaining proportionally
        # Total ratio parts: 60 + 30 + 30 = 120
        train_extra = int(math.ceil(remaining_classes * 0.6))  # 60% of remaining
        test_extra = int(remaining_classes * 0.3)    # 30% of remaining
        val_extra = int(remaining_classes * 0.3)    # 30% of remaining
        
        # Calculate final splits
        train_size = min_classes + train_extra
        val_size = min_classes + val_extra 
        test_size = min_classes + test_extra
    else:
        train_size = min_classes 
        val_size = min_classes 
        test_size = min_classes 
    
    
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

