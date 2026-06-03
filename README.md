# Deep Learning based Link Prediction for Minimizing Maximum Betweenness

## Overview

This project implements a simplified Graph Neural Network (GNN) framework for predicting missing edges in a network that can reduce the **maximum betweenness centrality** of the graph.

The model is inspired by the research paper:

**“Deep Learning based Link Prediction for Minimizing Maximum Betweenness”**

The implementation uses:

* Pure PyTorch
* Manual GraphSAGE-style neighborhood aggregation
* MLP-based edge classification
* Edge ranking using Triplet Margin Loss

This version does not require PyTorch Geometric.

---

# Problem Statement

Given a graph:

[
G = (V,E)
]

Find a missing edge:

[
e \notin E
]

such that adding the edge minimizes the graph’s:

[
\max_{v \in V} BC(v)
]

where:

* (BC(v)) is the betweenness centrality of node (v)
* the objective is to reduce bottlenecks and traffic congestion inside the network

---

# Architecture

The model contains three main components:

## 1. GraphSAGE-style GNN Encoder

Learns node embeddings using neighborhood aggregation.

### Aggregation

[
h_{N(v)}^{(k)} =
\frac{1}{|N(v)|}
\sum_{u \in N(v)}
h_u^{(k-1)}
]

### Update Rule

[
h_v^{(k)} =
\sigma(
W_1 h_v^{(k-1)}
+
W_2 h_{N(v)}^{(k)}
)
]

---

## 2. Edge Classifier (MLP1)

Predicts whether a missing edge is important or not.

Output:

* 1 → useful edge
* 0 → unimportant edge

Uses:

* Binary Cross Entropy Loss

---

## 3. Edge Ranker (MLP2)

Assigns ranking scores to candidate edges.

Higher score:

* better edge for minimizing maximum betweenness

Uses:

* Triplet Margin Ranking Loss

---

# Triplet Margin Loss

For three edges:

[
(m,n,o)
]

the loss is:

[
Loss =
\max(0,-(s_m-s_n)+M)
]

where:

* (s_m,s_n,s_o) are predicted ranking scores
* (M) is margin

This helps the model learn relative ordering between edges.

---

# Features

* Pure PyTorch implementation
* No PyTorch Geometric required
* GraphSAGE-style aggregation
* Edge classification + ranking
* Triplet loss based ranking optimization
* Easy to extend for real-world graphs

---

# Project Structure

```bash
project/
│
├── model.py
├── train.py
├── README.md
└── requirements.txt
```

---

# Requirements

Install dependencies:

```bash
pip install torch
```

---

# Training

Run training:

```bash
python train.py
```

---

# Example Workflow

1. Generate graph
2. Create adjacency matrix
3. Generate node features
4. Sample candidate missing edges
5. Train GNN encoder
6. Rank candidate edges
7. Select best edge

---

# Hyperparameters

| Parameter           | Value  |
| ------------------- | ------ |
| Hidden Dimension    | 32     |
| Learning Rate       | 0.0085 |
| Dropout             | 0.5    |
| Epochs              | 50     |
| Edge Embedding Size | 64     |

---

# Dataset

Synthetic graphs can be generated using:

* Barabási–Albert Networks
* Erdős–Rényi Networks
* Trees
* Path Graphs

---

# Evaluation Metrics

The original paper evaluates performance using:

* mAP@k
* Pearson Correlation
* Spearman Correlation
* Kendall Tau Correlation

---

# Applications

* Transportation networks
* Social networks
* Communication systems
* Congestion reduction
* Infrastructure optimization
* Critical path analysis

---

# Future Improvements

* Add real-world datasets
* Use Graph Attention Networks (GAT)
* Implement dynamic graphs
* Add NetworkX integration
* GPU optimization
* Distributed training

---

# References

1. GraphSAGE:
   Hamilton et al., NIPS 2017

2. Betweenness Centrality:
   Freeman, 1977

3. GNN Centrality Estimation:
   Maurya et al., TKDD 2021

4. Brandes Algorithm:
   Brandes, 2001

---
