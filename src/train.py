import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch.func import functional_call  # used for functional API 

from utils import get_ego_subgraph, get_structural_subgraph, contrastive_loss


def compute_ho_loss_functional(fmodel, params, buffers, data, task_indices, device, return_pred=False):
    ce_criterion = nn.CrossEntropyLoss()
    
    labels, preds = [], []
    loss = 0.0 
    task_count = 0
    
    # Process each task
    for task_nodes in task_indices:
        task_count += 1

        # Process each node in the current task
        for idx in task_nodes:
            # Create label mapping
            all_labels = data.y[task_nodes].tolist()
            task_labels = sorted(set(all_labels))
            task_label_to_local_idx = {label: i for i, label in enumerate(task_labels)}
          
            idx_item = idx.item() if torch.is_tensor(idx) else idx

            _, ego_features, ego_edge_index = get_ego_subgraph(idx, data, num_hops=2, max_nodes=50)
            _, cut_features, cut_edge_index = get_structural_subgraph(idx, data, strategy='high_degree')


            node_embedding = functional_call(
                fmodel, (params, buffers), (ego_features.to(device), ego_edge_index.to(device), cut_features.to(device), cut_edge_index.to(device))
            )
            
            # Handle label mapping (removed duplicate code)
            original_label = data.y[idx_item].item()
            if task_label_to_local_idx is not None:
                label = task_label_to_local_idx[original_label]
            else:
                label = original_label
            
            labels.append(label)

            ce_loss = ce_criterion(node_embedding, torch.tensor([label], device=device))
            loss += ce_loss

            # For evaluation, compute predictions
            if return_pred:
                pred = node_embedding.argmax().item()
                preds.append(pred)
    
    if return_pred:
        return loss/task_count, preds, labels

    return loss/task_count

# Compute the loss (cross-entropy + contrastive) using functional model call
def compute_cl_loss_functional(fmodel, params, buffers, dataset_name, data, task_indices, config, device, return_pred=False):
    """
    Compute classification and contrastive loss for a batch of task indices.
    
    Args:
        fmodel: The model function
        params: Model parameters
        buffers: Model buffers
        dataset_name: Name of dataset for config lookup
        data: Graph data object
        task_indices: List of tensors, each containing node indices for a task
        config: Configuration dictionary
        device: Device to run on
        task_label_to_local_idx: Mapping from original labels to task-local labels
        return_pred: Whether to return predictions (for evaluation)
    """
    ce_criterion = torch.nn.CrossEntropyLoss()
    
    labels = []
    preds = []
    
    loss = torch.tensor(0.0, device=device)
    task_count = 0
    # Process each task
    for task_nodes in task_indices:
        task_count +=1
        # Process each node in the current task
        for idx in task_nodes:
            # Create label mapping
            all_labels = data.y[task_nodes].tolist()
            task_labels = sorted(set(all_labels))
            task_label_to_local_idx = {label: i for i, label in enumerate(task_labels)}

            idx_item = idx.item() if torch.is_tensor(idx) else idx

            # Get ego subgraph for the node
            target_subgraph_idx, ego_features, ego_edge_index = get_ego_subgraph(idx_item, data)
            
            # Forward pass through model
            node_embedding, graph_embedding = functional_call(
                fmodel, (params, buffers), 
                (ego_features.to(device), ego_edge_index.to(device))
            )
            
            # Get embedding for target node
            target_node_emb = node_embedding[target_subgraph_idx]
            
            # Handle label mapping
            original_label = data.y[idx_item].item()
            if task_label_to_local_idx is not None:
                label = task_label_to_local_idx[original_label]
            else:
                label = original_label
            
            labels.append(label)

            ce_loss = ce_criterion(target_node_emb, torch.tensor(label, device=device))

            # For evaluation, compute predictions
            if return_pred:
                pred = target_node_emb.argmax().item()
                preds.append(pred)
                continue  # Skip contrastive loss computation during evaluation
            
            # Contrastive learning setup
            if graph_embedding is not None:
                graph_embedding = F.normalize(graph_embedding, p=2, dim=-1)
                
                # Find positive samples: same class, different node
                same_class_mask = data.y[task_nodes] == data.y[idx_item]
                positive_indices = task_nodes[same_class_mask].tolist()
                
                # Remove current node from positives
                if idx_item in positive_indices:
                    positive_indices.remove(idx_item)
                
                if len(positive_indices) == 0:
                    continue  # No positive samples available
                
                # Pick a random positive example
                positive_idx = random.choice(positive_indices)
                _, pos_ego_features, pos_ego_edge_index = get_ego_subgraph(positive_idx, data)
                _, pos_graph_embedding = functional_call(
                    fmodel, (params, buffers),
                    (pos_ego_features.to(device), pos_ego_edge_index.to(device))
                )
                pos_graph_embedding = F.normalize(pos_graph_embedding, p=2, dim=-1)
                
                # Find negative samples: different class
                diff_class_mask = data.y[task_nodes] != data.y[idx_item]
                negative_indices = task_nodes[diff_class_mask]
                
                if len(negative_indices) == 0:
                    continue  # No negative samples available
                
                # Hard Negative Mining
                max_neg_samples = config[dataset_name]['max_negative_samples']
                n_neg = min(len(negative_indices), max_neg_samples)
                
                # Batch compute negative embeddings
                negative_graph_embeddings = []
                for neg_idx in negative_indices[:n_neg]:
                    neg_idx_item = neg_idx.item() if torch.is_tensor(neg_idx) else neg_idx
                    neg_subgraph_idx, neg_ego_features, neg_ego_edge_index = get_ego_subgraph(neg_idx_item, data)
                    _, neg_graph_embedding = functional_call(
                        fmodel, (params, buffers),
                        (neg_ego_features.to(device), neg_ego_edge_index.to(device))
                    )
                    neg_graph_embedding = F.normalize(neg_graph_embedding, p=2, dim=-1)
                    negative_graph_embeddings.append(neg_graph_embedding)
                
                if len(negative_graph_embeddings) == 0:
                    continue
                
                negative_graph_embeddings = torch.vstack(negative_graph_embeddings)
                
                # Ensure proper dimensions
                if graph_embedding.dim() == 1:
                    graph_embedding = graph_embedding.unsqueeze(0)
                if pos_graph_embedding.dim() == 1:
                    pos_graph_embedding = pos_graph_embedding.unsqueeze(0)
                
                # Hard negative mining: select hardest negatives
                sim_vec = F.cosine_similarity(graph_embedding, negative_graph_embeddings, dim=1)
                min_neg_samples = config[dataset_name]['min_negative_samples']
                k = min(min_neg_samples, sim_vec.size(0))
                
                if k > 0:
                    topk_sim, topk_idx = torch.topk(sim_vec, k=k, largest=True)
                    hardest_negatives = negative_graph_embeddings[topk_idx]
                    
                    # Expand anchor and positive to match number of negatives
                    anchor_expanded = graph_embedding.expand(k, -1)
                    positive_expanded = pos_graph_embedding.expand(k, -1)
                    
                    cl_loss = contrastive_loss(
                            anchor_expanded, positive_expanded, hardest_negatives, 
                            float(config[dataset_name]['temperature'])
                        )
                else:
                    cl_loss  = torch.tensor(0.0, device=device)
            
            loss += ce_loss + cl_loss
    
    if return_pred:
        return loss, preds, labels

    return loss/task_count


def meta_training_step(model, dataset_name, data, tasks, optimizer, config, device):
    """
    Perform one meta-training step with proper MAML implementation.
    
    Args:
        model: The neural network model
        dataset_name: Name of dataset for config lookup
        data: Graph data object
        tasks: List of task dictionaries with 'support' and 'query' keys
        optimizer: Meta-optimizer
        config: Configuration dictionary
        device: Device to run on
    """
    model.train()
        
    # Extract support and query indices
    support_indices = [task['support'] for task in tasks]
    query_indices = [task['query'] for task in tasks]
    
    
    # Get model parameters and buffers
    params = OrderedDict(model.named_parameters())
    buffers = OrderedDict(model.named_buffers())
    
    # Inner loop parameters
    lr_inner = config[dataset_name]['lr_inner']
    inner_steps = config[dataset_name]['inner_steps']
    
    # Copy parameters for fast adaptation
    fast_params = OrderedDict((name, param.clone()) for name, param in params.items())
    
    task_support_loss = 0.0
    
    # Inner loop: Fast adaptation on support sets
    for step in range(inner_steps):
        # Compute loss on support set

        if config['model_type'] == "GNNEncoder":
            support_loss = compute_cl_loss_functional(
                model, fast_params, buffers, dataset_name, data, support_indices,
                config, device
            )
        else:
            support_loss = compute_ho_loss_functional(
                model, fast_params, buffers, data, 
                support_indices, device
            )


        # Compute gradients w.r.t. fast_params (enable second-order gradients)
        grads = torch.autograd.grad(
            support_loss, fast_params.values(), 
            create_graph=True, allow_unused=True, retain_graph=True
        )
        
        # Handle None gradients
        grads = [g if g is not None else torch.zeros_like(p) 
                for g, p in zip(grads, fast_params.values())]
        
        # Update fast parameters
        fast_params = OrderedDict([
            (name, param - lr_inner * grad)
            for (name, param), grad in zip(fast_params.items(), grads)
        ])
        
        task_support_loss += support_loss.item()
    
    task_support_loss /= max(1, inner_steps)
    
    # Outer loop: Meta-update using query set and adapted parameters
    if config['model_type'] == "GNNEncoder":
        query_loss = compute_cl_loss_functional(
            model, fast_params, buffers, dataset_name, data, query_indices,  
            config, device
        )
    else:
        query_loss =  compute_ho_loss_functional(
                model, fast_params, buffers, data, 
                query_indices, device
            )
    
    # Meta-update
    optimizer.zero_grad()
    query_loss.backward()
    optimizer.step()
    
    task_query_loss = query_loss.item()
    
    return task_support_loss, task_query_loss


def meta_test_step(model, dataset_name, data, tasks, config, device):
    """
    Perform meta-testing with proper adaptation and evaluation.
    """
    model.eval()
    
    # Extract support and query indices
    support_indices = [task['support'] for task in tasks]
    query_indices = [task['query'] for task in tasks]

    # Get model parameters
    params = OrderedDict(model.named_parameters())
    buffers = OrderedDict(model.named_buffers())
    
    # Inner loop parameters
    lr_inner = config[dataset_name]['lr_inner']
    inner_steps = config[dataset_name]['inner_steps']
    
    # Copy parameters for adaptation
    fast_params = OrderedDict((name, param.clone()) for name, param in params.items())
    
    task_support_loss = 0.0
    
    # Inner loop: Adapt on support set (no gradients needed for meta-parameters)
    for step in range(inner_steps):
        
        if config['model_type'] == "GNNEncoder":
            support_loss = compute_cl_loss_functional(
                model, fast_params, buffers, dataset_name, data, support_indices,
                config, device)
        else:
            support_loss = compute_ho_loss_functional(
                model, fast_params, buffers, data, 
                support_indices, device
            )

        # Compute gradients w.r.t. fast_params (no second-order gradients needed)
        grads = torch.autograd.grad(
            support_loss, fast_params.values(),
            create_graph=False, allow_unused=True
        )
        
        grads = [g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, fast_params.values())]
        
        # Update fast parameters
        fast_params = OrderedDict([
            (name, param - lr_inner * grad)
            for (name, param), grad in zip(fast_params.items(), grads)
        ])
        
        task_support_loss += support_loss.item()
    
    task_support_loss /= max(1, inner_steps)
    
    # Evaluation on query set
    with torch.no_grad():

        if config['model_type'] == "GNNEncoder":
            query_loss, preds, true_labels = compute_cl_loss_functional(
                model, fast_params, buffers, dataset_name, data, query_indices,
                config, device, 
                return_pred=True
            )
        else:
            query_loss, preds, true_labels = compute_ho_loss_functional(
                model, fast_params, buffers, data, 
                support_indices, device, 
                return_pred=True
            )
    
    # Compute accuracy metrics
    correct = sum(p == l for p, l in zip(preds, true_labels))
    total = len(true_labels)
    overall_accuracy = correct / max(1, total)
    
    # Class-wise accuracy
    class_correct = {}
    class_total = {}
    
    for pred, label in zip(preds, true_labels):
        if label not in class_total:
            class_total[label] = 0
            class_correct[label] = 0
        
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    class_accuracy = {
        cls: class_correct[cls] / class_total[cls]
        for cls in class_total if class_total[cls] > 0
    }
    
    return query_loss.item(), overall_accuracy, class_accuracy, preds, true_labels


# Train across a batch of tasks
def train(model, dataset_name, data, batch_tasks, optimizer, config, device):
    """Wrapper for training step."""
    return meta_training_step(model, dataset_name, data, batch_tasks, optimizer, config, device)


# Evaluate across multiple meta-test tasks
def evaluate(model, dataset_name, data, batch_tasks, config, device):
    """Wrapper for evaluation step."""
    loss, acc, class_accuracy, predictions, labels = meta_test_step(
        model, dataset_name, data, batch_tasks, config, device
    )
    return loss, acc, class_accuracy