# Graph Meta-Learning with Contrastive Supervision

This project explores **meta-learning for graph-structured data** by combining **node classification** and **contrastive learning** within a MAML-style optimization framework. It is designed to improve generalization across tasks and enable more robust graph representation learning.


## Objective

The goal is to train a graph neural network (GNN) that:

- Learns transferable representations from small subgraphs.
- Combines **cross-entropy loss** (for classification) with **triplet contrastive loss** (to shape the embedding space).
- Utilizes **hard negative mining** to sharpen the contrastive signal.
- Adopts **Model-Agnostic Meta-Learning (MAML)** to simulate few-shot learning conditions over multiple graph tasks.


## Key Concepts

### Graph Neural Networks (GNNs)
A GNN learns node and graph-level embeddings by passing messages between connected nodes. Here, it outputs:

- **Node embeddings** (used for classification).
- **Graph-level embeddings** (used for contrastive learning).

### Meta-Learning (MAML)
The model is trained across **multiple small tasks**, where each task involves:

1. **Support set** (used for adaptation).
2. **Query set** (used to compute meta-gradients).

This meta-learning loop helps the GNN adapt quickly to new tasks with few samples.
The model is explicitly **trained to adapt**. In **standard supervised learning**, the model learns to solve a specific task by minimizing a loss function over that task’s data. But in **MAML**, the model is **not trained to solve a task directly** — it's trained so that it can **learn new tasks quickly** with only a few gradient steps.

MAML optimizes the **initial parameters** $\theta$ so that after a small number of updates, the model performs well on **any task** from the task distribution.

This involves two loops:

#### Inner loop (task adaptation):

* For each task $T_i$, take a few gradient descent steps on a support set (e.g., training data of task $T_i$) to get **adapted parameters** $\theta'_i$.

#### Outer loop (meta-update):

* Evaluate the adapted model $f_{\theta'_i}$ on the **query set** 
* Use the loss on this query set to update the original $\theta$, so that next time, it adapts better.

So the meta-objective is:

$$
\min_{\theta} \sum_{T_i \sim p(T)} \mathcal{L}_{T_i}(f_{\theta'_i}) = \mathcal{L}_{T_i}(f_{\theta - \alpha \nabla_\theta \mathcal{L}_{T_i}^{\text{support}}})
$$

### Classification Loss
Standard cross-entropy loss is used to train the model to predict node labels correctly.

### Contrastive Learning
To encourage meaningful embeddings, a **triplet loss** is added:

- **Anchor**: Subgraph around the query node.
- **Positive**: Another subgraph from the same class.
- **Negative**: Subgraph from a different class.

Hard negatives are selected based on similarity to the anchor (using cosine distance), improving the quality of contrastive signals.



## How to Run

### Installation

Make sure you have Python 3.8+ and install dependencies via:

```bash
pip install -r requirements.txt
```

### Training

```bash
python src/main.py
```

This will train the model using MAML-style episodes with classification and contrastive losses.

### Evaluation

Validation and test performance are computed during training. You can adjust task sampling, contrastive loss weight, and hard negative settings in the configuration file.



## Project Structure

```
src/
├── train.py          # Main training loop
├── meta_learning.py  # Inner/outer loop logic
├── model.py          # GNN and functional forward
├── utils.py          # Subgraph extraction, sampling
...
requirements.txt      # Python dependencies
README.md             # This file
```


## Acknowledgments

This work builds on ideas from:

* MAML: Model-Agnostic Meta-Learning
* Graph contrastive learning literature
* Node classification benchmarks in PyTorch Geometric
