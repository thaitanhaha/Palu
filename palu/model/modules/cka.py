import torch
import torch.nn as nn


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
    dev: torch.device,
):
    W = raw_linear.weight.detach().to(dev)

    out_features, in_features = raw_linear.out_features, raw_linear.in_features
    # TODO: fix head_dim
    head_dim = 64
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


def greedy_reorder_from_cka(S: torch.Tensor, group_size: int = 4):
    n = S.size(0)
    S = S.clone()

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

        i, j = best_pair

        group = [i, j]
        used.add(i)
        used.add(j)

        S[i, :] = -1
        S[:, i] = -1
        S[j, :] = -1
        S[:, j] = -1

        groups.append(group)

    remaining = [k for k in range(n) if k not in used]

    for h in remaining:
        best_g = None
        best_score = -1

        for gi, g in enumerate(groups):
            if len(g) >= group_size:
                continue

            score = S[h, g].mean().item()

            if score > best_score:
                best_score = score
                best_g = gi

        groups[best_g].append(h)
        used.add(h)

    perm = []
    for g in groups:
        perm.extend(g)

    return perm


def invert_perm(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def reorder_linear_weight(raw_linear: torch.nn.Linear, cka_scores):
    # TODO: fix head_dim
    head_dim = 64

    perm = greedy_reorder_from_cka(cka_scores, group_size = 4)

    W = raw_linear.weight.data
    n_heads = W.size(0) // head_dim

    heads = W.view(n_heads, head_dim, -1)
    heads = heads[perm]

    raw_linear.weight.data = heads.reshape_as(W)
    return raw_linear







