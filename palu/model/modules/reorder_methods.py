import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import defaultdict


def invert_perm(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic = torch.norm(X.T @ Y, p="fro").pow(2)
    var_x = torch.norm(X.T @ X, p="fro")
    var_y = torch.norm(Y.T @ Y, p="fro")

    return hsic / (var_x * var_y + eps)

@torch.no_grad()
def compute_cka_for_linear(
    raw_linear: nn.Linear,
    head_dim: int,
    dev: torch.device,
):
    W = raw_linear.weight.detach().to(dev)

    out_features, in_features = raw_linear.out_features, raw_linear.in_features
    num_heads = raw_linear.out_features // head_dim

    heads = W.view(num_heads, head_dim, in_features)

    cka_scores = torch.zeros(
        num_heads,
        num_heads,
        device=dev,
        dtype=W.dtype,
    )

    for i in range(num_heads):
        Xi = heads[i].T

        for j in range(i, num_heads):
            Yj = heads[j].T

            score = _linear_cka(Xi, Yj)

            cka_scores[i, j] = score
            cka_scores[j, i] = score
            
    return cka_scores


def greedy_reorder_based_on_cka(S_original: torch.Tensor, group_size: int = 4):
    n = S_original.size(0)
    S = S_original.clone()

    S.fill_diagonal_(-1)

    used = set()
    groups = []

    while len(used) < n:
        best_val = -1
        best_pair = None

        for i in range(n):
            if i in used:
                continue
            for j in range(n):
                if j in used or i == j:
                    continue
                if S[i, j] > best_val:
                    best_val = S[i, j]
                    best_pair = (i, j)

        if best_pair is None:
            remaining = [h for h in range(n) if h not in used]
            if remaining:
                groups.append(remaining)
                used.update(remaining)
            break

        i, j = best_pair
        group = [i, j]
        used.update(group)

        while len(group) < group_size and len(used) < n:
            best_h = None
            best_score = -float('inf')

            for h in range(n):
                if h in used:
                    continue

                score = S[h, group].mean().item()

                if score > best_score:
                    best_score = score
                    best_h = h

            if best_h is not None:
                group.append(best_h)
                used.add(best_h)
            else:
                break

        groups.append(group)

    perm = [h for g in groups for h in g]
    return perm

@torch.no_grad()
def reorder_cka_static(
    raw_linear: torch.nn.Linear, 
    num_group: int, 
    head_dim: int,
    dev: torch.device
):
    W = raw_linear.weight.data
    n_heads = W.size(0) // head_dim

    group_size = n_heads // num_group

    cka_scores = compute_cka_for_linear(raw_linear, head_dim, dev)

    perm = greedy_reorder_based_on_cka(cka_scores, group_size)
    inv_perm = invert_perm(perm)

    group_to_heads = defaultdict(list)
    for i, item in enumerate(perm):
        group_to_heads[i // group_size].append(item)

    heads = W.view(n_heads, head_dim, -1)
    heads = heads[perm]

    raw_linear.weight.data = heads.reshape_as(W)

    if raw_linear.bias is not None:
        bias = raw_linear.bias.data.view(n_heads, head_dim)
        bias = bias[perm]
        raw_linear.bias.data = bias.reshape_as(raw_linear.bias.data)

    return raw_linear, group_to_heads, inv_perm


@torch.no_grad()
def cluster_labels_based_on_cka(
    cka_scores: torch.Tensor,
):
    cka_np = cka_scores.detach().cpu().numpy()

    distance_matrix = 1.0 - cka_np

    best_group_labels = -1
    best_score = -1

    max_clusters = min(10, cka_scores.shape[0])
    for n_clusters in range(2, max_clusters):
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
        group_labels = clustering.fit_predict(distance_matrix)

        score = silhouette_score(distance_matrix, group_labels, metric="precomputed")

        if score > best_score:
            best_score = score
            best_group_labels = group_labels
    
    return best_group_labels

@torch.no_grad()
def reorder_cka_dynamic(
    raw_linear: nn.Linear,
    head_dim: int,
    dev: torch.device,
):
    W = raw_linear.weight.data
    n_heads = W.size(0) // head_dim

    cka_scores = compute_cka_for_linear(raw_linear, head_dim, dev)

    group_labels = cluster_labels_based_on_cka(cka_scores)

    group_to_heads = defaultdict(list)
    for i, g in enumerate(group_labels):
        group_to_heads[g].append(i)

    perm = []
    for g in sorted(group_to_heads.keys()):
        perm.extend(group_to_heads[g])

    inv_perm = invert_perm(perm)

    heads = W.view(n_heads, head_dim, -1)
    heads = heads[perm]

    raw_linear.weight.data = heads.reshape_as(W)

    if raw_linear.bias is not None:
        bias = raw_linear.bias.data.view(n_heads, head_dim)
        bias = bias[perm]
        raw_linear.bias.data = bias.reshape_as(raw_linear.bias.data)

    return raw_linear, group_to_heads, inv_perm


@torch.no_grad()
def cluster_labels_based_on_histogram(
    histograms
):
    histograms = histograms / (histograms.sum(dim=1, keepdim=True) + 1e-8)
    head_hist_np = histograms.cpu().numpy()

    num_heads = head_hist_np.shape[0]
    distance_matrix = np.zeros((num_heads, num_heads), dtype=np.float32)

    for h1 in range(num_heads):
        for h2 in range(h1, num_heads):
            dist = wasserstein_distance(head_hist_np[h1], head_hist_np[h2])
            distance_matrix[h1, h2] = dist
            distance_matrix[h2, h1] = dist

    best_group_labels = -1
    best_score = -1

    max_clusters = min(10, num_heads)
    for n_clusters in range(2, max_clusters):
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
        group_labels = clustering.fit_predict(distance_matrix)

        score = silhouette_score(distance_matrix, group_labels, metric="precomputed")

        if score > best_score:
            best_score = score
            best_group_labels = group_labels
    
    return best_group_labels

@torch.no_grad()
def reorder_histogram_dynamic(
    raw_linear: torch.nn.Linear, 
    histograms,
    head_dim: int,
    dev: torch.device
):
    group_labels = cluster_labels_based_on_histogram(histograms)

    group_to_heads = defaultdict(list)
    for i, g in enumerate(group_labels):
        group_to_heads[g].append(i)

    perm = []
    for g in sorted(group_to_heads.keys()):
        perm.extend(group_to_heads[g])

    inv_perm = invert_perm(perm)

    W = raw_linear.weight.data
    n_heads = W.size(0) // head_dim

    heads = W.view(n_heads, head_dim, -1)
    heads = heads[perm]

    raw_linear.weight.data = heads.reshape_as(W)

    if raw_linear.bias is not None:
        bias = raw_linear.bias.data.view(n_heads, head_dim)
        bias = bias[perm]
        raw_linear.bias.data = bias.reshape_as(raw_linear.bias.data)

    return raw_linear, group_to_heads, inv_perm





