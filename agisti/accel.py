"""
Rust-accelerated backend — transparent bridge to agisti-core.

Tries to import the Rust-compiled agisti_core module.
If available, exports fast versions of CKA, integrity, stats, LoRA.
If not available (Rust not compiled), falls back to pure Python.

Usage:
    from agisti.accel import rust_available, fast_cka_all_pairs, fast_sha256

The caller never needs to know whether Rust is loaded or not.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ─── Try importing Rust module ────────────────────────

rust_available = False

try:
    import agisti_core as _rs  # type: ignore[import-not-found]

    rust_available = True
    logger.info("agisti-core Rust backend loaded — performance mode ON")
except ImportError:
    _rs = None  # type: ignore[assignment]
    logger.info(
        "agisti-core Rust backend not found — using pure Python fallback. "
        "To enable: cd agisti-core && maturin develop --release"
    )


# ═══════════════════════════════════════════════════════
#  CKA
# ═══════════════════════════════════════════════════════

def fast_cka_pair(x, y, debiased: bool = True) -> float:
    """CKA between two activation matrices. Rust-accelerated if available."""
    if rust_available:
        import numpy as np

        x_np = x.cpu().numpy().astype(np.float32) if hasattr(x, "cpu") else x
        y_np = y.cpu().numpy().astype(np.float32) if hasattr(y, "cpu") else y
        return _rs.compute_cka_pair(x_np, y_np, debiased)

    # Fallback: pure Python/PyTorch
    from agisti.utils.math_ops import compute_cka

    return compute_cka(x, y, debiased=debiased)


def fast_cka_all_pairs(
    target_acts: list,
    ref_acts: list,
    debiased: bool = True,
) -> Any:
    """
    CKA for all (target_layer, ref_layer) pairs.

    Rust: rayon-parallelised, GIL released. ~10-50x faster.
    Python fallback: nested for-loop.

    Returns: 2D numpy array of shape (n_target, n_ref).
    """
    if rust_available:
        import numpy as np

        t_np = [
            a.cpu().numpy().astype(np.float32) if hasattr(a, "cpu") else a
            for a in target_acts
        ]
        r_np = [
            a.cpu().numpy().astype(np.float32) if hasattr(a, "cpu") else a
            for a in ref_acts
        ]
        return _rs.compute_cka_all_pairs(t_np, r_np, debiased)

    # Fallback
    import numpy as np
    from agisti.utils.math_ops import compute_cka

    n_t = len(target_acts)
    n_r = len(ref_acts)
    result = np.zeros((n_t, n_r), dtype=np.float64)
    for i, ta in enumerate(target_acts):
        for j, ra in enumerate(ref_acts):
            result[i, j] = compute_cka(ta, ra, debiased=debiased)
    return result


# ═══════════════════════════════════════════════════════
#  Integrity Hashing
# ═══════════════════════════════════════════════════════

def fast_sha256(data: bytes) -> str:
    """SHA-256 hex digest. Rust ring if available, else hashlib."""
    if rust_available:
        import numpy as np

        arr = np.frombuffer(data, dtype=np.uint8)
        return _rs.sha256_bytes(arr)

    import hashlib

    return hashlib.sha256(data).hexdigest()


def fast_parallel_sha256(buffers: list[bytes]) -> list[str]:
    """Multi-threaded SHA-256 for a list of byte buffers."""
    if rust_available:
        import numpy as np

        np_bufs = [np.frombuffer(b, dtype=np.uint8) for b in buffers]
        return _rs.parallel_sha256(np_bufs)

    import hashlib

    return [hashlib.sha256(b).hexdigest() for b in buffers]


def fast_fingerprint(flat_array) -> str:
    """Fast norm+mean+first fingerprint from a flat f32 array."""
    if rust_available:
        import numpy as np

        arr = flat_array if isinstance(flat_array, np.ndarray) else flat_array.cpu().numpy()
        return _rs.fast_fingerprint(arr.astype(np.float32).ravel())

    # Fallback
    import numpy as np

    arr = flat_array if isinstance(flat_array, np.ndarray) else flat_array.cpu().numpy()
    arr = arr.ravel().astype(np.float32)
    if len(arr) == 0:
        return "empty"
    norm = float(np.linalg.norm(arr))
    mean = float(arr.mean())
    first = float(arr[0])
    return f"{norm:.10e}|{mean:.10e}|{first:.10e}|{len(arr)}"


# ═══════════════════════════════════════════════════════
#  Statistics
# ═══════════════════════════════════════════════════════

def fast_comb(n: int, k: int) -> int:
    """Binomial coefficient C(n,k)."""
    if rust_available:
        return _rs.comb(n, k)
    import math
    return math.comb(n, k)


def fast_binomial_pmf(n: int, k: int, p: float) -> float:
    """Binomial PMF."""
    if rust_available:
        return _rs.binomial_pmf(n, k, p)
    return fast_comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def fast_chi2_survival(x: float, df: int = 1) -> float:
    """Chi-squared survival function."""
    if rust_available:
        return _rs.chi2_survival(x, df)
    import math
    if x <= 0:
        return 1.0
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    # Fallback to Python implementation in mcnemar.py
    from agisti.benchmark.mcnemar import _chi2_survival

    return _chi2_survival(x, df)


def fast_welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-test. Returns (t_stat, p_value)."""
    if rust_available:
        return _rs.welch_t_test(a, b)
    from agisti.utils.math_ops import welch_t_test

    return welch_t_test(a, b)


# ═══════════════════════════════════════════════════════
#  LoRA
# ═══════════════════════════════════════════════════════

def fast_parallel_norms(a_list: list, b_list: list) -> list[float]:
    """Compute ||A@B||_F for all LoRA layer pairs in parallel."""
    if rust_available:
        import numpy as np

        a_np = [
            a.cpu().numpy().astype(np.float32) if hasattr(a, "cpu") else a
            for a in a_list
        ]
        b_np = [
            b.cpu().numpy().astype(np.float32) if hasattr(b, "cpu") else b
            for b in b_list
        ]
        return _rs.parallel_norms(a_np, b_np)

    # Fallback
    import torch

    norms = []
    for a, b in zip(a_list, b_list):
        full = a @ b
        norms.append(torch.norm(full).item())
    return norms


def fast_budget_check(norms: list[float], budget: float) -> tuple[float, bool]:
    """Check total LoRA norm vs budget. Returns (total, within_budget)."""
    if rust_available:
        return _rs.budget_check(norms, budget)
    import math
    total = math.sqrt(sum(n * n for n in norms))
    return total, total <= budget
