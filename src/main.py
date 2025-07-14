# Install required packages
import torch
import random
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import os
import time

from torch.optim import Adam

from torch_geometric.datasets import Planetoid, CoraFull
from torch_geometric.utils import to_undirected

from utils import create_tasks, create_train_valid_test_split
from models import GNNEncoder
from train import train, evaluate

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Description: Script to run our model.')
    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--n_way', type=int, help='n way', default=2)
    parser.add_argument('--k_shot', type=int, help='k shot', default=3)
    return parser

def main(dataset_name, seed, device):
    # Start timing
    main_start_time = time.time()
    
    # Set random seeds for reproducibility
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(seed)

    # Load and preprocess dataset
    if dataset_name == 'Cora':
        dataset = Planetoid(root='../data/' + dataset_name, name=dataset_name)
    elif dataset_name == 'CiteSeer':
        dataset = Planetoid(root='../data/' + dataset_name, name=dataset_name)
    elif dataset_name == 'CoraFull':
        dataset = CoraFull(root='../data/' + dataset_name)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    data = dataset[0].to(device)
    data.edge_index = to_undirected(data.edge_index)
    
    # Configuration
    config = {
        ##2-3, 2-5, 3-3, 3-5
        'n_way': 5,  # number of classes per task
        'k_shot': 1,  # number of support samples per class
        'n_query': 5,  # number of query samples per class
        'n_epochs': 10000,
        'model_type': 'GNNEncoder', # or 'HighOrderGNNEncoder'
        'batch_size': 10,  # Reduced for memory efficiency
        "CoraFull": {  # 70 classes in total
            'input_dim': dataset.num_features,
            'hidden_dim': 128,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'n_train_tasks': 1000,
            'n_val_tasks': 200,
            'n_test_tasks': 400,
            "max_negative_samples": 6,
            "min_negative_samples": 3,
            "inner_steps": 20, 
            "lr_inner": 0.01,  
            "patience": 10,
            "temperature": 0.1,
        },
        "Cora": {  # 7 classes in total
            'input_dim': dataset.num_features,
            'hidden_dim': 128,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'n_train_tasks': 100,
            'n_val_tasks': 20,
            'n_test_tasks': 40,
            "max_negative_samples": 6,
            "min_negative_samples": 3,
            "inner_steps": 5,
            "lr_inner": 0.01,
            "patience": 10,
            "temperature": 0.1,
        },
        "CiteSeer": {  # 6 classes in total
            'input_dim': dataset.num_features,
            'hidden_dim': 128,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'n_train_tasks': 100,
            'n_val_tasks': 20,
            'n_test_tasks': 40,
            "max_negative_samples": 6,
            "min_negative_samples": 3,
            "inner_steps": 5,
            "lr_inner": 0.01,
            "patience": 10,
            "temperature": 0.1,
        }
    }

    # Split classes into train, validation, and test
    train_classes, val_classes, test_classes = create_train_valid_test_split(
        data, seed,
        config['n_way']
    )

    print(f'Dataset: {dataset_name}')
    print(f'Total classes: {dataset.num_classes}')
    print(f'Training Classes: {train_classes}')
    print(f'Validation Classes: {val_classes}')
    print(f'Testing Classes: {test_classes}')
    
    # Create tasks
    print("Creating tasks...")
    train_tasks = create_tasks(
        data, train_classes,
        n_way=config['n_way'],
        k_shot=config['k_shot'],
        n_query=config['n_query'],
        n_tasks=config[dataset_name]['n_train_tasks']
    )

    val_tasks = create_tasks(
        data, val_classes,
        n_way=config['n_way'],
        k_shot=config['k_shot'],
        n_query=config['n_query'],
        n_tasks=config[dataset_name]['n_val_tasks']
    )

    test_tasks = create_tasks(
        data, test_classes,
        n_way=config['n_way'],
        k_shot=config['k_shot'],
        n_query=config['n_query'],
        n_tasks=config[dataset_name]['n_test_tasks']
    )

    print(f"Created {len(train_tasks)} train tasks, {len(val_tasks)} val tasks, {len(test_tasks)} test tasks")


    # Initialize model and optimizer
    print("Initializing GNNEncoder...")
    model = GNNEncoder(
        config[dataset_name]['input_dim'],
        config[dataset_name]['hidden_dim'],
        config['n_way'],
    ).to(device)
    
    optimizer = Adam(
        model.parameters(),
        lr=config[dataset_name]['learning_rate'],
        weight_decay=config[dataset_name]['weight_decay']
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Starting meta-training...")

    # Training tracking
    best_val_acc = -1.0
    epochs_no_improve = 0
    best_model_state = None
    
    # Track training history
    training_history = {
        'epoch': [],
        'support_loss': [],
        'query_loss': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }

    # Meta-training loop with progress bar
    training_start_time = time.time()
    epoch_pbar = tqdm(range(config['n_epochs']), desc="Training Epochs")
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        model.train()
        epoch_support_loss = 0.0
        epoch_query_loss = 0.0
        n_batches = 0

        # Process training tasks in batches with progress bar
        batch_size = config['batch_size']
        batch_pbar = tqdm(range(0, len(train_tasks), batch_size), 
                         desc=f"Epoch {epoch+1} Batches", 
                         leave=False)
        
        for i in batch_pbar:
            batch_tasks = train_tasks[i:i + batch_size]
            
            try:
                batch_support_loss, batch_query_loss = train(
                    model, dataset_name, data, batch_tasks, optimizer, config, device
                )
                epoch_support_loss += batch_support_loss
                epoch_query_loss += batch_query_loss
                n_batches += 1
                
                # Update batch progress bar
                batch_pbar.set_postfix({
                    'support_loss': f'{batch_support_loss:.4f}',
                    'query_loss': f'{batch_query_loss:.4f}'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in batch {i//batch_size}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        # Average losses
        epoch_support_loss /= max(1, n_batches)
        epoch_query_loss /= max(1, n_batches)

        # Validation
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{config['n_epochs']}")
            print(f"Train - Support Loss: {epoch_support_loss:.4f}, Query Loss: {epoch_query_loss:.4f}")

            # Validate with progress bar
            model.eval()
            total_val_loss = 0.0
            total_val_acc = 0.0
            n_val_batches = 0

            val_batch_size = min(len(val_tasks), config['batch_size'])
            val_pbar = tqdm(range(0, len(val_tasks), val_batch_size), 
                           desc="Validation", 
                           leave=False)
            
            for i in val_pbar:
                batch_tasks = val_tasks[i:i + val_batch_size]
                
                try:
                    loss, acc, _ = evaluate(model, dataset_name, data, batch_tasks, config, device)
                    total_val_loss += loss
                    total_val_acc += acc
                    n_val_batches += 1
                    
                    val_pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{acc:.4f}'})
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM error in validation batch {i//val_batch_size}, skipping...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

            # Average validation metrics
            avg_val_loss = total_val_loss / max(1, n_val_batches)
            avg_val_acc = total_val_acc / max(1, n_val_batches)

            print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.4f}")
            
            # Save training history
            training_history['epoch'].append(epoch + 1)
            training_history['support_loss'].append(float(epoch_support_loss))
            training_history['query_loss'].append(float(epoch_query_loss))
            training_history['val_loss'].append(float(avg_val_loss))
            training_history['val_acc'].append(float(avg_val_acc))
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            training_history['epoch_time'].append(float(epoch_time))
            print(f"Epoch time: {epoch_time:.2f} seconds")

            # Early stopping and model saving
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                epochs_no_improve = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
                # Save best model
                model_path = f'./models/{dataset_name}_{config['n_way']}_way_{config['k_shot']}_shot_best_model.pth'
                os.makedirs('./models/', exist_ok=True)
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= config[dataset_name]['patience']:
                print(f"Early stopping at epoch {epoch+1} due to no improvement.")
                break
                
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'support_loss': f'{epoch_support_loss:.4f}',
            'query_loss': f'{epoch_query_loss:.4f}'
        })
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    print(f"\nTotal training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")

    # Load best model for testing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
    else:
        print("Warning: No best model saved, using current model state.")

    # Meta-testing with progress bar
    print("\nStarting meta-testing...")
    test_start_time = time.time()
    model.eval()
    total_test_loss = 0.0
    total_test_acc = 0.0
    n_test_batches = 0

    test_batch_size = min(len(test_tasks), config['batch_size'])
    test_pbar = tqdm(range(0, len(test_tasks), test_batch_size), 
                     desc="Testing")
    
    for i in test_pbar:
        batch_tasks = test_tasks[i:i + test_batch_size]
        
        try:
            loss, acc, class_acc = evaluate(model, dataset_name, data, batch_tasks, config, device)
            total_test_loss += loss
            total_test_acc += acc
            n_test_batches += 1
            
            test_pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{acc:.4f}'})
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM error in test batch {i//test_batch_size}, skipping...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    # Final test results
    avg_test_loss = total_test_loss / max(1, n_test_batches)
    avg_test_acc = total_test_acc / max(1, n_test_batches)
    
    # Calculate test time
    test_time = time.time() - test_start_time
    print(f"\nTest time: {test_time:.2f} seconds")

    # Calculate total time
    total_time = time.time() - main_start_time
    
    print(f"\nFinal Test Results:")
    print(f"Meta-Test Loss: {avg_test_loss:.4f}")
    print(f"Meta-Test Accuracy: {avg_test_acc:.4f}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("=" * 50)

    # Add timing info to training history
    training_history['total_training_time'] = float(total_training_time)
    training_history['test_time'] = float(test_time)
    training_history['total_time'] = float(total_time)

    return avg_test_acc, training_history 


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_parser()
    try:
        args = parser.parse_args()
        dataset_name = args.dataset
        n_way = args.n_way
        k_shot = args.k_shot
    except SystemExit:
        print("No command line arguments provided, using default values.")

    seed = 42
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test on multiple datasets
    datasets = ['CoraFull']  # Add 'CiteSeer', 'CoraFull' as needed
    runs = 3
    
    # Initialize results dictionary
    all_results = {
        'experiment_info': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(device),
            'seed': seed,
            'runs': runs
        },
        'results': {},
        'total_experiment_time': 0.0
    }
    
    # Start experiment timer
    experiment_start_time = time.time()
    
    for dataset_name in datasets:
        print(f"\n{'='*20} {dataset_name} {'='*20}")
        
        dataset_results = {
            'test_accuracies': [],
            'mean_accuracy': 0.0,
            'std_accuracy': 0.0,
            'training_histories': [],
            'run_times': [],
            'total_dataset_time': 0.0
        }
        
        dataset_start_time = time.time()
        
        try:
            for run in range(runs):
                print(f"\n--- Run {run + 1}/{runs} ---")
                run_start_time = time.time()
                
                test_acc, training_history = main(dataset_name, seed + run, device)
                
                run_time = time.time() - run_start_time
                print(f"{dataset_name} Run {run + 1} test accuracy: {test_acc:.4f}")
                print(f"Run {run + 1} total time: {run_time:.2f} seconds ({run_time/60:.2f} minutes)")
                
                dataset_results['test_accuracies'].append(float(test_acc))
                dataset_results['training_histories'].append(training_history)
                dataset_results['run_times'].append(float(run_time))
            
            # Calculate statistics
            accuracies = np.array(dataset_results['test_accuracies'])
            dataset_results['mean_accuracy'] = float(np.mean(accuracies))
            dataset_results['std_accuracy'] = float(np.std(accuracies))
            
            # Calculate total dataset time
            dataset_results['total_dataset_time'] = float(time.time() - dataset_start_time)
            
            print(f"\n{dataset_name} Summary:")
            print(f"Mean accuracy: {dataset_results['mean_accuracy']:.4f} ± {dataset_results['std_accuracy']:.4f}")
            print(f"All accuracies: {dataset_results['test_accuracies']}")
            print(f"Total dataset time: {dataset_results['total_dataset_time']:.2f} seconds ({dataset_results['total_dataset_time']/60:.2f} minutes)")
            print(f"Average run time: {np.mean(dataset_results['run_times']):.2f} seconds")
            
            all_results['results'][dataset_name] = dataset_results

        except Exception as e:
            print(f"Error running {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            all_results['results'][dataset_name] = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'partial_time': float(time.time() - dataset_start_time)
            }
    
    # Calculate total experiment time
    all_results['total_experiment_time'] = float(time.time() - experiment_start_time)
    print(f"\nTotal experiment time: {all_results['total_experiment_time']:.2f} seconds ({all_results['total_experiment_time']/60:.2f} minutes)")

    # Save results to JSON file
    os.makedirs('./results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_filename = f'./results/{dataset_name}_meta_learning_results_{timestamp}.json'
    
    with open(results_filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_filename}")
    
    # Also save a summary file
    summary = {
        'timestamp': all_results['experiment_info']['timestamp'],
        'total_experiment_time': all_results['total_experiment_time'],
        'summary': {}
    }
    
    for dataset, results in all_results['results'].items():
        if 'mean_accuracy' in results:
            summary['summary'][dataset] = {
                'mean_accuracy': results['mean_accuracy'],
                'std_accuracy': results['std_accuracy'],
                'runs': len(results['test_accuracies']),
                'total_dataset_time': results['total_dataset_time'],
                'average_run_time': np.mean(results['run_times'])
            }
    
    summary_filename = f'./results/{dataset_name}_meta_learning_summary_{timestamp}.json'
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_filename}")