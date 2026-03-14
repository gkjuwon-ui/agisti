"""
Tests for agisti.utils.math_ops — SVD, CKA, Procrustes, statistics.
"""

from __future__ import annotations

import math
import pytest
import torch

from agisti.utils.math_ops import (
    truncated_svd,
    reconstruct_from_svd,
    low_rank_approximation,
    explained_variance_ratio,
    adaptive_rank,
    factorize_lora,
    compute_cka,
    _standard_cka,
    _debiased_cka,
    procrustes_alignment,
    procrustes_similarity,
    cosine_similarity_matrix,
    cosine_distance,
    gradient_norm,
    parameter_norm,
    welch_t_test,
    exponential_moving_average,
    linear_regression_slope,
    entropy,
    kl_divergence,
)


class TestTruncatedSVD:
    """Tests for SVD utilities."""

    def test_shapes(self):
        M = torch.randn(100, 50)
        U, S, Vt = truncated_svd(M, rank=10)
        assert U.shape == (100, 10)
        assert S.shape == (10,)
        assert Vt.shape == (10, 50)

    def test_rank_clamped(self):
        """Rank should be clamped to min dimension."""
        M = torch.randn(10, 5)
        U, S, Vt = truncated_svd(M, rank=100)
        assert S.shape[0] <= 5

    def test_reconstruction(self):
        M = torch.randn(50, 30)
        U, S, Vt = truncated_svd(M, rank=30)
        M_hat = reconstruct_from_svd(U, S, Vt)
        assert torch.allclose(M, M_hat, atol=1e-4)

    def test_low_rank_approximation(self):
        M = torch.randn(50, 30)
        M_lr = low_rank_approximation(M, rank=5)
        assert M_lr.shape == M.shape
        # Low rank should have rank <= 5
        rank = torch.linalg.matrix_rank(M_lr).item()
        assert rank <= 5

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="Expected 2D"):
            truncated_svd(torch.randn(3, 4, 5), rank=2)


class TestExplainedVariance:
    """Tests for explained variance ratio."""

    def test_full_rank(self):
        M = torch.randn(20, 10)
        ratio = explained_variance_ratio(M, rank=10)
        assert ratio == pytest.approx(1.0, abs=1e-5)

    def test_partial_rank(self):
        M = torch.randn(50, 30)
        ratio = explained_variance_ratio(M, rank=5)
        assert 0 < ratio < 1

    def test_monotonic(self):
        """Increasing rank should increase explained variance."""
        M = torch.randn(50, 30)
        r1 = explained_variance_ratio(M, rank=1)
        r5 = explained_variance_ratio(M, rank=5)
        r10 = explained_variance_ratio(M, rank=10)
        assert r1 <= r5 <= r10


class TestAdaptiveRank:
    """Tests for adaptive rank selection."""

    def test_returns_reasonable(self):
        M = torch.randn(50, 20)
        k = adaptive_rank(M, target_variance=0.95)
        assert 1 <= k <= 20

    def test_higher_target_needs_higher_rank(self):
        M = torch.randn(50, 20)
        k90 = adaptive_rank(M, target_variance=0.90)
        k99 = adaptive_rank(M, target_variance=0.99)
        assert k90 <= k99

    def test_max_rank_clamp(self):
        M = torch.randn(50, 20)
        k = adaptive_rank(M, target_variance=0.999, max_rank=5)
        assert k <= 5


class TestFactorizeLora:
    """Tests for LoRA factorization."""

    def test_shapes(self):
        delta = torch.randn(128, 64)
        A, B = factorize_lora(delta, rank=8)
        assert A.shape == (8, 64)
        assert B.shape == (128, 8)

    def test_reconstruction_quality(self):
        delta = torch.randn(64, 32)
        A, B = factorize_lora(delta, rank=32)
        reconstructed = B @ A
        # Full rank should be near-perfect
        assert torch.allclose(delta, reconstructed, atol=1e-4)

    def test_low_rank_approximation(self):
        delta = torch.randn(100, 50)
        A, B = factorize_lora(delta, rank=5)
        reconstructed = B @ A
        assert reconstructed.shape == delta.shape


class TestCKA:
    """Tests for CKA computation."""

    def test_identical_representations(self):
        """CKA of identical representations should be ~1."""
        X = torch.randn(50, 20)
        cka = compute_cka(X, X)
        assert cka == pytest.approx(1.0, abs=0.05)

    def test_orthogonal_representations(self):
        """CKA of orthogonal representations should be ~0."""
        n = 100
        X = torch.randn(n, 20)
        # Make Y orthogonal to X
        Y = torch.randn(n, 20)
        # Project out X components
        Q, _ = torch.linalg.qr(X)
        Y_orth = Y - Q @ (Q.T @ Y)
        cka = compute_cka(X, Y_orth)
        assert cka < 0.3  # Should be small

    def test_cka_range(self):
        """CKA should be in [0, 1]."""
        X = torch.randn(50, 10)
        Y = torch.randn(50, 15)
        cka = compute_cka(X, Y)
        assert 0.0 <= cka <= 1.0

    def test_standard_vs_debiased(self):
        """Both versions should give similar results for large n."""
        X = torch.randn(200, 20)
        Y = torch.randn(200, 30)
        standard = _standard_cka(X, Y)
        debiased = _debiased_cka(X, Y)
        # Should be similar for large samples
        assert abs(standard - debiased) < 0.2

    def test_mismatched_samples_raises(self):
        X = torch.randn(50, 10)
        Y = torch.randn(60, 10)
        with pytest.raises(ValueError, match="same number"):
            compute_cka(X, Y)

    def test_different_dimensions_ok(self):
        """CKA should work with different feature dimensions."""
        X = torch.randn(50, 10)
        Y = torch.randn(50, 30)
        cka = compute_cka(X, Y)
        assert 0.0 <= cka <= 1.0


class TestProcrustes:
    """Tests for Procrustes alignment."""

    def test_rotation_recovery(self):
        """Should recover a known rotation."""
        n, d = 50, 10
        source = torch.randn(n, d)

        # Apply known rotation
        R_true = torch.linalg.qr(torch.randn(d, d))[0]
        target = source @ R_true

        R_found, residual = procrustes_alignment(source, target)

        # Residual should be small
        assert residual < 1e-3

        # R_found should be close to R_true
        assert torch.allclose(R_found, R_true, atol=1e-3)

    def test_identity(self):
        """Same source and target → identity rotation, zero residual."""
        X = torch.randn(30, 5)
        R, residual = procrustes_alignment(X, X)
        assert residual < 1e-4
        assert torch.allclose(R, torch.eye(5), atol=1e-3)

    def test_similarity_perfect(self):
        """Identical matrices should have similarity ~1."""
        X = torch.randn(30, 5)
        sim = procrustes_similarity(X, X)
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_similarity_range(self):
        X = torch.randn(30, 5)
        Y = torch.randn(30, 5)
        sim = procrustes_similarity(X, Y)
        assert 0.0 <= sim <= 1.0

    def test_shape_mismatch_raises(self):
        X = torch.randn(30, 5)
        Y = torch.randn(30, 10)
        with pytest.raises(ValueError, match="Shape mismatch"):
            procrustes_alignment(X, Y)


class TestCosineSimilarity:
    """Tests for cosine similarity utilities."""

    def test_self_similarity(self):
        X = torch.randn(10, 5)
        sim = cosine_similarity_matrix(X)
        assert sim.shape == (10, 10)
        # Diagonal should be ~1
        for i in range(10):
            assert sim[i, i] == pytest.approx(1.0, abs=1e-5)

    def test_cross_similarity_shape(self):
        X = torch.randn(10, 5)
        Y = torch.randn(20, 5)
        sim = cosine_similarity_matrix(X, Y)
        assert sim.shape == (10, 20)

    def test_cosine_distance_same(self):
        v = torch.randn(10)
        assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-5)

    def test_cosine_distance_opposite(self):
        v = torch.randn(10)
        assert cosine_distance(v, -v) == pytest.approx(2.0, abs=1e-4)


class TestGradientNorm:
    """Tests for gradient/parameter norm computation."""

    def test_gradient_norm_with_grads(self):
        model = torch.nn.Linear(10, 5)
        x = torch.randn(3, 10)
        y = model(x)
        y.sum().backward()

        norm = gradient_norm(model)
        assert norm > 0

    def test_gradient_norm_no_grads(self):
        model = torch.nn.Linear(10, 5)
        norm = gradient_norm(model)
        assert norm == 0.0

    def test_parameter_norm(self):
        model = torch.nn.Linear(10, 5)
        norm = parameter_norm(model)
        assert norm > 0


class TestWelchTTest:
    """Tests for Welch's t-test."""

    def test_identical_samples(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        t, p = welch_t_test(a, a)
        assert abs(t) < 0.01
        assert p > 0.9

    def test_different_samples(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [10.0, 11.0, 12.0, 13.0, 14.0]
        t, p = welch_t_test(a, b)
        assert abs(t) > 2
        assert p < 0.05

    def test_insufficient_samples(self):
        a = [1.0]
        b = [2.0]
        t, p = welch_t_test(a, b)
        assert p == 1.0


class TestEMA:
    """Tests for exponential moving average."""

    def test_empty(self):
        assert exponential_moving_average([]) == []

    def test_single(self):
        assert exponential_moving_average([5.0]) == [5.0]

    def test_smoothing(self):
        values = [1.0, 10.0, 1.0, 10.0, 1.0]
        ema = exponential_moving_average(values, alpha=0.5)
        assert len(ema) == 5
        # EMA should be smoother than original
        assert max(ema) < max(values)

    def test_convergence(self):
        """EMA of constant should be constant."""
        values = [5.0] * 10
        ema = exponential_moving_average(values, alpha=0.3)
        for v in ema:
            assert v == pytest.approx(5.0)


class TestLinearRegressionSlope:
    """Tests for linear regression slope."""

    def test_increasing(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        slope = linear_regression_slope(values)
        assert slope == pytest.approx(1.0, abs=0.01)

    def test_decreasing(self):
        values = [5.0, 4.0, 3.0, 2.0, 1.0]
        slope = linear_regression_slope(values)
        assert slope == pytest.approx(-1.0, abs=0.01)

    def test_constant(self):
        values = [3.0, 3.0, 3.0, 3.0]
        slope = linear_regression_slope(values)
        assert slope == pytest.approx(0.0, abs=0.01)

    def test_single_value(self):
        assert linear_regression_slope([1.0]) == 0.0

    def test_empty(self):
        assert linear_regression_slope([]) == 0.0


class TestEntropy:
    """Tests for Shannon entropy."""

    def test_uniform(self):
        """Uniform distribution should have maximum entropy."""
        n = 4
        probs = [1.0 / n] * n
        h = entropy(probs)
        assert h == pytest.approx(math.log(n), abs=1e-6)

    def test_deterministic(self):
        """Deterministic distribution should have 0 entropy."""
        probs = [1.0, 0.0, 0.0, 0.0]
        h = entropy(probs)
        assert h == pytest.approx(0.0, abs=1e-6)

    def test_binary(self):
        probs = [0.5, 0.5]
        h = entropy(probs)
        assert h == pytest.approx(math.log(2), abs=1e-6)


class TestKLDivergence:
    """Tests for KL divergence."""

    def test_same_distribution(self):
        p = [0.3, 0.7]
        kl = kl_divergence(p, p)
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_positive(self):
        p = [0.3, 0.7]
        q = [0.5, 0.5]
        kl = kl_divergence(p, q)
        assert kl > 0

    def test_infinite_when_q_zero(self):
        p = [0.5, 0.5]
        q = [1.0, 0.0]
        kl = kl_divergence(p, q)
        assert kl == float("inf")

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            kl_divergence([0.5, 0.5], [0.3, 0.3, 0.4])
