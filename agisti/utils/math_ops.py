"""
Math Operations — linear algebra and statistical utilities
used throughout the AGISTI pipeline.

Includes:
- SVD helpers for LoRA factorization
- CKA (Centered Kernel Alignment) computation
- Procrustes alignment
- Statistical tests & metrics
- Cosine similarity helpers
- Gradient norm computation

All operations use PyTorch tensors unless otherwise noted.
"""

from __future__ import annotations

import math
from typing import overload

import torch
from torch import Tensor


# ─── SVD Utilities ────────────────────────────────────

def truncated_svd(
    matrix: Tensor,
    rank: int,
    driver: str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute truncated SVD of a matrix.

    Returns U[:, :rank], S[:rank], Vt[:rank, :] such that
    matrix ≈ U @ diag(S) @ Vt.

    Args:
        matrix: 2D tensor to factorize.
        rank: Number of singular values to keep.
        driver: SVD driver ('gesvd', 'gesvda', None=auto).

    Returns:
        (U, S, Vt) truncated to specified rank.
    """
    if matrix.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {matrix.dim()}D")

    rank = min(rank, min(matrix.shape))

    U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
    return U[:, :rank], S[:rank], Vt[:rank, :]


def reconstruct_from_svd(
    U: Tensor,
    S: Tensor,
    Vt: Tensor,
) -> Tensor:
    """
    Reconstruct matrix from SVD components: U @ diag(S) @ Vt.
    """
    return U @ torch.diag(S) @ Vt


def low_rank_approximation(
    matrix: Tensor,
    rank: int,
) -> Tensor:
    """
    Compute rank-k approximation of a matrix via SVD.
    """
    U, S, Vt = truncated_svd(matrix, rank)
    return reconstruct_from_svd(U, S, Vt)


def explained_variance_ratio(
    matrix: Tensor,
    rank: int,
) -> float:
    """
    Fraction of total variance explained by top-k singular values.

    Returns value in [0, 1]. Higher = better rank-k approximation.
    """
    _, S_full, _ = torch.linalg.svd(matrix, full_matrices=False)
    total = (S_full ** 2).sum()
    if total < 1e-10:
        return 1.0
    top_k_var = (S_full[:rank] ** 2).sum()
    return (top_k_var / total).item()


def adaptive_rank(
    matrix: Tensor,
    target_variance: float = 0.95,
    max_rank: int = 64,
) -> int:
    """
    Find minimum rank to explain target fraction of variance.

    Args:
        matrix: 2D tensor.
        target_variance: Fraction of variance to explain (0-1).
        max_rank: Maximum rank to return.

    Returns:
        Minimum rank k such that top-k SVs explain ≥ target_variance.
    """
    _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
    total = (S ** 2).sum()
    if total < 1e-10:
        return 1

    cumsum = torch.cumsum(S ** 2, dim=0) / total
    for k in range(len(cumsum)):
        if cumsum[k].item() >= target_variance:
            return min(k + 1, max_rank)

    return min(len(S), max_rank)


def factorize_lora(
    delta: Tensor,
    rank: int,
) -> tuple[Tensor, Tensor]:
    """
    Factorize a weight delta into LoRA format: A, B where delta ≈ B @ A.

    Convention: B is (out_dim, rank), A is (rank, in_dim).

    Args:
        delta: Weight matrix delta (out_dim × in_dim).
        rank: LoRA rank.

    Returns:
        (A, B) tensors.
    """
    U, S, Vt = truncated_svd(delta, rank)
    # B = U * sqrt(S), A = sqrt(S) * Vt
    sqrt_S = torch.sqrt(S)
    B = U * sqrt_S.unsqueeze(0)     # (out_dim, rank)
    A = sqrt_S.unsqueeze(1) * Vt    # (rank, in_dim)
    return A, B


# ─── CKA (Centered Kernel Alignment) ─────────────────

def compute_cka(
    X: Tensor,
    Y: Tensor,
    debiased: bool = True,
) -> float:
    """
    Compute CKA between two activation matrices.

    CKA measures representational similarity between
    two sets of activations (e.g., from different layers
    or different models).

    Args:
        X: Activation matrix (n_samples × dim_x).
        Y: Activation matrix (n_samples × dim_y).
        debiased: Whether to use debiased CKA.

    Returns:
        CKA similarity in [0, 1].
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have same number of samples: "
            f"{X.shape[0]} vs {Y.shape[0]}"
        )

    if debiased:
        return _debiased_cka(X, Y)
    return _standard_cka(X, Y)


def _standard_cka(X: Tensor, Y: Tensor) -> float:
    """Standard (biased) CKA."""
    K = X @ X.T
    L = Y @ Y.T

    # Center
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - 1.0 / n
    K_c = H @ K @ H
    L_c = H @ L @ H

    # HSIC
    hsic_xy = (K_c * L_c).sum() / ((n - 1) ** 2)
    hsic_xx = (K_c * K_c).sum() / ((n - 1) ** 2)
    hsic_yy = (L_c * L_c).sum() / ((n - 1) ** 2)

    denom = math.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0

    return (hsic_xy / denom).item()


def _debiased_cka(X: Tensor, Y: Tensor) -> float:
    """
    Debiased CKA from Song et al. (2012).

    More accurate for finite samples.
    """
    K = X @ X.T
    L = Y @ Y.T

    hsic_xy = _debiased_hsic(K, L)
    hsic_xx = _debiased_hsic(K, K)
    hsic_yy = _debiased_hsic(L, L)

    denom = math.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0

    return max(0.0, min(1.0, hsic_xy / denom))


def _debiased_hsic(K: Tensor, L: Tensor) -> float:
    """
    Debiased HSIC estimator.

    Ref: Song et al., "Feature selection via dependence maximization", 2012.
    """
    n = K.shape[0]
    if n < 4:
        return 0.0

    # Zero diagonal
    K_tilde = K.clone()
    L_tilde = L.clone()
    K_tilde.fill_diagonal_(0.0)
    L_tilde.fill_diagonal_(0.0)

    # Main term
    term1 = (K_tilde * L_tilde).sum()

    # Correction terms
    k_sum = K_tilde.sum()
    l_sum = L_tilde.sum()
    term2 = k_sum * l_sum / ((n - 1) * (n - 2))

    # Cross term
    term3 = 2.0 * (K_tilde.sum(dim=1) @ L_tilde.sum(dim=1))

    result = (term1 + term2 - term3) / (n * (n - 3))
    return result.item()


# ─── Procrustes Alignment ────────────────────────────

def procrustes_alignment(
    source: Tensor,
    target: Tensor,
) -> tuple[Tensor, float]:
    """
    Orthogonal Procrustes alignment.

    Finds the orthogonal matrix R that minimizes
    ||target - source @ R||_F.

    Args:
        source: Source matrix (n × d).
        target: Target matrix (n × d).

    Returns:
        (R, residual) where R is the optimal rotation/reflection
        and residual is the Frobenius norm of the error.
    """
    if source.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: {source.shape} vs {target.shape}"
        )

    M = source.T @ target
    U, _, Vt = torch.linalg.svd(M)

    # Optimal rotation
    R = U @ Vt

    # Handle reflection: ensure det(R) > 0
    if torch.det(R) < 0:
        # Flip sign of last column of U
        U_fixed = U.clone()
        U_fixed[:, -1] *= -1
        R = U_fixed @ Vt

    residual = torch.norm(target - source @ R, p="fro").item()
    return R, residual


def procrustes_similarity(
    source: Tensor,
    target: Tensor,
) -> float:
    """
    Procrustes similarity (1 - normalized residual).

    Returns value in [0, 1] where 1 = identical (up to rotation).
    """
    _, residual = procrustes_alignment(source, target)
    norm = torch.norm(target, p="fro").item()
    if norm < 1e-10:
        return 1.0 if residual < 1e-10 else 0.0
    return max(0.0, 1.0 - residual / norm)


# ─── Cosine Similarity ───────────────────────────────

def cosine_similarity_matrix(
    X: Tensor,
    Y: Tensor | None = None,
) -> Tensor:
    """
    Pairwise cosine similarity matrix.

    Args:
        X: (n × d) tensor.
        Y: (m × d) tensor. If None, computes self-similarity of X.

    Returns:
        (n × m) cosine similarity matrix.
    """
    X_norm = torch.nn.functional.normalize(X, p=2, dim=1)
    if Y is None:
        return X_norm @ X_norm.T
    Y_norm = torch.nn.functional.normalize(Y, p=2, dim=1)
    return X_norm @ Y_norm.T


def cosine_distance(a: Tensor, b: Tensor) -> float:
    """Cosine distance between two vectors (1 - cosine_similarity)."""
    sim = torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0),
    )
    return (1.0 - sim.item())


# ─── Gradient Utilities ──────────────────────────────

def gradient_norm(
    parameters: list[Tensor] | torch.nn.Module,
    norm_type: float = 2.0,
) -> float:
    """
    Compute gradient norm across parameters.

    Args:
        parameters: Model or list of parameters.
        norm_type: Type of norm (default L2).

    Returns:
        Total gradient norm.
    """
    if isinstance(parameters, torch.nn.Module):
        params = [p for p in parameters.parameters() if p.grad is not None]
    else:
        params = [p for p in parameters if p.grad is not None]

    if not params:
        return 0.0

    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach(), norm_type)
            for p in params
        ]),
        norm_type,
    )
    return total_norm.item()


def parameter_norm(
    model: torch.nn.Module,
    norm_type: float = 2.0,
) -> float:
    """Compute total parameter norm."""
    norms = [
        torch.norm(p.detach(), norm_type)
        for p in model.parameters()
    ]
    if not norms:
        return 0.0
    return torch.norm(torch.stack(norms), norm_type).item()


# ─── Statistical Utilities ───────────────────────────

def welch_t_test(
    a: list[float],
    b: list[float],
) -> tuple[float, float]:
    """
    Welch's t-test for unequal variances.

    Args:
        a: First sample.
        b: Second sample.

    Returns:
        (t_statistic, p_value_approx).
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0, 1.0

    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-12:
        return 0.0, 1.0

    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    den = (
        (var_a / n_a) ** 2 / (n_a - 1)
        + (var_b / n_b) ** 2 / (n_b - 1)
    )
    if den < 1e-12:
        df = n_a + n_b - 2
    else:
        df = num / den

    # Approximate p-value using normal approximation for large df
    p_value = 2.0 * _normal_cdf(-abs(t_stat))

    return t_stat, p_value


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def exponential_moving_average(
    values: list[float],
    alpha: float = 0.1,
) -> list[float]:
    """
    Compute EMA of a sequence.

    Args:
        values: Input sequence.
        alpha: Smoothing factor (0 < alpha ≤ 1).

    Returns:
        EMA sequence.
    """
    if not values:
        return []

    result = [values[0]]
    for v in values[1:]:
        ema = alpha * v + (1 - alpha) * result[-1]
        result.append(ema)
    return result


def linear_regression_slope(
    values: list[float],
) -> float:
    """
    Compute OLS slope for evenly-spaced data.

    Args:
        values: Sequence of y-values (x assumed to be 0, 1, 2, ...).

    Returns:
        Slope of the linear fit.
    """
    n = len(values)
    if n < 2:
        return 0.0

    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))

    if abs(den) < 1e-10:
        return 0.0
    return num / den


def entropy(probs: list[float] | Tensor) -> float:
    """
    Shannon entropy (nats).

    Args:
        probs: Probability distribution (must sum to ~1).

    Returns:
        Entropy in nats.
    """
    if isinstance(probs, Tensor):
        probs = probs.tolist()

    h = 0.0
    for p in probs:
        if p > 1e-10:
            h -= p * math.log(p)
    return h


def kl_divergence(
    p: list[float],
    q: list[float],
) -> float:
    """
    KL divergence D_KL(P || Q).

    Args:
        p: True distribution.
        q: Approximate distribution.

    Returns:
        KL divergence (nats). Returns inf if q has zeros where p doesn't.
    """
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")

    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-10:
            if qi < 1e-10:
                return float("inf")
            kl += pi * math.log(pi / qi)
    return kl
