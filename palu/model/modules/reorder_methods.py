import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import defaultdict


def invert_perm(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic = torch.norm(X.T @ Y, p="fro").pow(2)
    var_x = torch.norm(X.T @ X, p="fro")
    var_y = torch.norm(Y.T @ Y, p="fro")

    score = hsic / (var_x * var_y + eps)

    if torch.isnan(score) or torch.isinf(score):
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)
        
    return score


@torch.no_grad()
def compute_cka_for_linear(raw_linear: nn.Linear, head_dim: int, dev: torch.device):
    W = raw_linear.weight.detach().to(dev)
    num_heads = raw_linear.out_features // head_dim

    heads = W.view(num_heads, head_dim, raw_linear.in_features)
    cka_scores = torch.zeros(num_heads, num_heads, device=dev, dtype=W.dtype)

    for i in range(num_heads):
        Xi = heads[i].T
        for j in range(i, num_heads):
            Yj = heads[j].T
            score = _linear_cka(Xi, Yj)
            cka_scores[i, j] = score
            cka_scores[j, i] = score
            
    return cka_scores


@torch.no_grad()
def greedy_reorder_based_on_cka(S_original: torch.Tensor, group_size: int = 4):
    assert group_size % 2 == 0, "group_size must be even."

    n = S_original.size(0)
    S = S_original.clone()

    S.fill_diagonal_(-1)

    used = set()
    pairs = []
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

        i, j = best_pair
        used.add(i)
        used.add(j)

        S[i, :] = -1
        S[:, i] = -1
        S[j, :] = -1
        S[:, j] = -1

        pairs.append([i, j])

    groups = []
    pairs_per_group = group_size // 2
    for i in range(0, len(pairs), pairs_per_group):
        group = []
        for pair in pairs[i:i + pairs_per_group]:
            group.extend(pair)
        groups.append(group)

    perm = [h for g in groups for h in g]
    return perm


def _find_best_clusters(distance_matrix):
    best_group_labels = -1
    best_score = -1

    max_clusters = distance_matrix.shape[0] // 2
    min_clusters = distance_matrix.shape[0] // 4 + 1
    for n_clusters in range(min_clusters, max_clusters + 1):
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
        group_labels = clustering.fit_predict(distance_matrix)

        score = silhouette_score(distance_matrix, group_labels, metric="precomputed")

        if score > best_score:
            best_score = score
            best_group_labels = group_labels
    
    return best_group_labels


def _find_best_clusters(distance_matrix):
    best_group_labels = -1
    best_score = -float('inf')

    max_clusters = distance_matrix.shape[0] // 2
    min_clusters = distance_matrix.shape[0] // 4 + 1
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
        group_labels = clustering.fit_predict(distance_matrix)

        base_score = silhouette_score(distance_matrix, group_labels, metric="precomputed")
        unique_labels, counts = np.unique(group_labels, return_counts=True)
        
        penalty = 0
        for count in counts:
            if count == 1:
                penalty += 0.3
            if count > 8:
                penalty += 0.2
            if count % 2 != 0:
                penalty += 0.05

        total_score = base_score - penalty

        if total_score > best_score:
            best_score = total_score
            best_group_labels = group_labels
    
    return best_group_labels


def _find_best_clusters_via_svd(distance_matrix, heads_flat, R_budget):
    best_group_labels = -1
    min_total_error = float('inf')

    max_clusters = distance_matrix.shape[0] // 2
    min_clusters = distance_matrix.shape[0] // 4 + 1

    for n_clusters in range(min_clusters, max_clusters + 1):
        # TODO: maybe uniform? neu uniform thi chi xet group co 2 3 4 duoc thoi?
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
        group_labels = clustering.fit_predict(distance_matrix)
        
        unique_labels = np.unique(group_labels)
        
        cluster_svd_data = {}
        total_energy_all_clusters = 0
        
        for label in unique_labels:
            head_indices = np.where(group_labels == label)[0]
            W_cluster = np.concatenate([heads_flat[idx] for idx in head_indices], axis=0)
            
            U, S, Vt = np.linalg.svd(W_cluster, full_matrices=False)
            
            cluster_energy = np.sum(S ** 2)
            total_energy_all_clusters += cluster_energy
            
            cluster_svd_data[label] = {"W_cluster": W_cluster, "U": U, "S": S, "Vt": Vt, "energy": cluster_energy}
        
        total_layer_error = 0.0
        used_rank_total = 0
        
        for label in unique_labels:
            data = cluster_svd_data[label]
            
            cluster_ratio = data["energy"] / total_energy_all_clusters
            target_rank = int(np.round(R_budget * cluster_ratio))
            
            max_possible_rank = min(data["W_cluster"].shape)
            target_rank = max(1, min(target_rank, max_possible_rank))
            
            used_rank_total += target_rank
            
            U_truncated = data["U"][:, :target_rank]
            S_truncated = np.diag(data["S"][:target_rank])
            Vt_truncated = data["Vt"][:target_rank, :]
            W_reconstructed = np.dot(U_truncated, np.dot(S_truncated, Vt_truncated))
            
            error = np.linalg.norm(data["W_cluster"] - W_reconstructed, 'fro') ** 2
            total_layer_error += error
            
        rank_penalty = abs(R_budget - used_rank_total) * 1e3 
        total_layer_error += rank_penalty

        if total_layer_error < min_total_error:
            min_total_error = total_layer_error
            best_group_labels = group_labels
            
    return best_group_labels



def _apply_permutation(raw_linear: nn.Linear, perm: list, head_dim: int):
    W = raw_linear.weight.data
    n_heads = W.size(0) // head_dim

    heads = W.view(n_heads, head_dim, -1)
    heads = heads[perm]
    raw_linear.weight.data = heads.reshape_as(W)

    if raw_linear.bias is not None:
        bias = raw_linear.bias.data.view(n_heads, head_dim)
        bias = bias[perm]
        raw_linear.bias.data = bias.reshape_as(raw_linear.bias.data)

    inv_perm = invert_perm(perm)
    return raw_linear, inv_perm


@torch.no_grad()
def reorder_cka_static(raw_linear: nn.Linear, num_group: int, head_dim: int, dev: torch.device):
    n_heads = raw_linear.weight.size(0) // head_dim
    group_size = n_heads // num_group

    cka_scores = compute_cka_for_linear(raw_linear, head_dim, dev)
    perm = greedy_reorder_based_on_cka(cka_scores, group_size)

    group_to_heads = defaultdict(list)
    for i, item in enumerate(perm):
        group_to_heads[i // group_size].append(item)

    raw_linear, inv_perm = _apply_permutation(raw_linear, perm, head_dim)
    return raw_linear, group_to_heads, inv_perm



@torch.no_grad()
def reorder_cka_dynamic(raw_linear: nn.Linear, head_dim: int, dev: torch.device):
    cka_scores = compute_cka_for_linear(raw_linear, head_dim, dev)
    distance_matrix = (1.0 - cka_scores).cpu().numpy()
    np.fill_diagonal(distance_matrix, 0)

    group_labels = _find_best_clusters(distance_matrix)

    group_to_heads = defaultdict(list)
    for i, g in enumerate(group_labels):
        group_to_heads[g].append(i)

    perm = []
    for g in sorted(group_to_heads.keys()):
        perm.extend(group_to_heads[g])

    raw_linear, inv_perm = _apply_permutation(raw_linear, perm, head_dim)
    return raw_linear, group_to_heads, inv_perm



@torch.no_grad()
def reorder_wasserstein_dynamic(raw_linear: nn.Linear, head_dim: int, dev: torch.device):
    W = raw_linear.weight.data
    n_heads = W.size(0) // head_dim
    heads_flat = W.view(n_heads, -1)

    sorted_heads = torch.sort(heads_flat, dim=1).values
    diff = sorted_heads[:, None, :] - sorted_heads[None, :, :]
    distance_matrix = diff.abs().mean(dim=-1).cpu().numpy()

    group_labels = _find_best_clusters(distance_matrix)

    group_to_heads = defaultdict(list)
    for i, g in enumerate(group_labels):
        group_to_heads[g].append(i)

    perm = []
    for g in sorted(group_to_heads.keys()):
        perm.extend(group_to_heads[g])

    raw_linear, inv_perm = _apply_permutation(raw_linear, perm, head_dim)
    return raw_linear, group_to_heads, inv_perm




