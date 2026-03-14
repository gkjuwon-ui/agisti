"""
McNemar's test — statistical significance testing for paired benchmarks.

Compares two paired binary classifiers (before/after surgery)
using McNemar's test. This determines if improvement is
statistically significant, not just random fluctuation.

Design: §6.1 — QuickBench significance testing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SignificanceResult:
    """Result of a significance test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    n_discordant: int  # pairs where results disagreed
    improved: int  # previously wrong, now correct
    degraded: int  # previously correct, now wrong
    confidence_interval: tuple[float, float] | None = None

    def summary(self) -> str:
        return (
            f"{self.test_name}: statistic={self.statistic:.4f}, "
            f"p={self.p_value:.4f}, "
            f"{'SIGNIFICANT' if self.significant else 'not significant'} "
            f"(α={self.alpha}), "
            f"+{self.improved}/-{self.degraded} changed"
        )


def mcnemar_test(
    before: list[bool],
    after: list[bool],
    alpha: float = 0.05,
    use_correction: bool = True,
) -> SignificanceResult:
    """
    McNemar's test for paired binary outcomes.

    Tests if the proportion of errors differs between two paired
    classifiers (before and after surgery).

    The contingency table:
                    After correct    After wrong
    Before correct:      a              b
    Before wrong:        c              d

    Only b and c (discordant pairs) matter for the test.
    b = degraded (was correct, now wrong)
    c = improved (was wrong, now correct)

    H0: b = c (no difference)
    H1: b ≠ c (significant difference)

    Args:
        before: List of boolean correctness for each problem (before surgery).
        after: List of boolean correctness for each problem (after surgery).
        alpha: Significance level (default 0.05).
        use_correction: Use continuity correction for small samples.

    Returns:
        SignificanceResult with test statistic, p-value, and significance.
    """
    if len(before) != len(after):
        raise ValueError(
            f"Paired data must have same length: "
            f"{len(before)} vs {len(after)}"
        )

    # Build contingency table
    a = b = c = d = 0
    for bef, aft in zip(before, after):
        if bef and aft:
            a += 1  # both correct
        elif bef and not aft:
            b += 1  # degraded
        elif not bef and aft:
            c += 1  # improved
        else:
            d += 1  # both wrong

    n_discordant = b + c

    if n_discordant == 0:
        return SignificanceResult(
            test_name="McNemar",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            n_discordant=0,
            improved=c,
            degraded=b,
        )

    # For small samples (n_discordant < 25), use exact binomial
    if n_discordant < 25:
        return _mcnemar_exact(b, c, alpha)

    # Chi-squared approximation
    if use_correction:
        # Edwards' continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        chi2 = (b - c) ** 2 / (b + c)

    # p-value from chi-squared distribution (1 df)
    p_value = _chi2_survival(chi2, df=1)

    # Wilson confidence interval for the proportion
    p_hat = c / (b + c) if (b + c) > 0 else 0.5
    ci = _wilson_ci(p_hat, b + c, alpha)

    return SignificanceResult(
        test_name="McNemar (χ²)",
        statistic=chi2,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        n_discordant=n_discordant,
        improved=c,
        degraded=b,
        confidence_interval=ci,
    )


def _mcnemar_exact(
    b: int,
    c: int,
    alpha: float,
) -> SignificanceResult:
    """Exact binomial test for small samples."""
    n = b + c
    if n == 0:
        return SignificanceResult(
            test_name="McNemar (exact)",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            n_discordant=0,
            improved=c,
            degraded=b,
        )

    # Two-tailed p-value: P(X ≤ min(b,c)) under Binomial(n, 0.5)
    k = min(b, c)
    p_value = 0.0
    for i in range(k + 1):
        p_value += _binomial_pmf(n, i, 0.5)
    p_value *= 2  # two-tailed
    p_value = min(1.0, p_value)

    return SignificanceResult(
        test_name="McNemar (exact binomial)",
        statistic=float(abs(b - c)),
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        n_discordant=n,
        improved=c,
        degraded=b,
    )


def _binomial_pmf(n: int, k: int, p: float) -> float:
    """Binomial probability mass function."""
    if k < 0 or k > n:
        return 0.0
    coeff = _comb(n, k)
    return coeff * (p ** k) * ((1 - p) ** (n - k))


def _comb(n: int, k: int) -> int:
    """Compute C(n, k) using multiplicative formula."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _chi2_survival(x: float, df: int = 1) -> float:
    """
    Survival function (1 - CDF) for chi-squared distribution.

    Uses the regularized incomplete gamma function.
    For df=1: P(X > x) = erfc(sqrt(x/2))
    """
    if x <= 0:
        return 1.0
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    # General case via incomplete gamma
    return _regularized_upper_gamma(df / 2.0, x / 2.0)


def _regularized_upper_gamma(a: float, x: float) -> float:
    """Upper regularized incomplete gamma function Q(a, x)."""
    # Use series expansion for small x, continued fraction for large x
    if x < a + 1:
        return 1.0 - _regularized_lower_gamma_series(a, x)
    return _upper_gamma_cf(a, x)


def _regularized_lower_gamma_series(a: float, x: float) -> float:
    """Lower regularized incomplete gamma via series expansion."""
    if x < 0:
        return 0.0
    term = 1.0 / a
    total = term
    for n in range(1, 200):
        term *= x / (a + n)
        total += term
        if abs(term) < 1e-15 * abs(total):
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _upper_gamma_cf(a: float, x: float) -> float:
    """Upper regularized incomplete gamma via continued fraction."""
    f = 1e-30
    c = 1e-30
    d = 1.0 / (x + 1.0 - a)
    h = d

    for i in range(1, 200):
        an = -i * (i - a)
        bn = x + 2.0 * i + 1.0 - a
        d = an * d + bn
        if abs(d) < 1e-30:
            d = 1e-30
        c = bn + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = c * d
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break

    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _wilson_ci(
    p_hat: float,
    n: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)

    # Z-score for confidence level
    z = _normal_ppf(1 - alpha / 2)
    z2 = z * z

    denominator = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denominator
    spread = z * math.sqrt(
        (p_hat * (1 - p_hat) + z2 / (4 * n)) / n
    ) / denominator

    return (max(0.0, center - spread), min(1.0, center + spread))


def _normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (percent-point function)."""
    # Rational approximation (Abramowitz & Stegun 26.2.23)
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    if p == 0.5:
        return 0.0

    if p < 0.5:
        sign = -1.0
        p = 1 - p
    else:
        sign = 1.0

    t = math.sqrt(-2.0 * math.log(1 - p))

    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    result = t - (c0 + c1 * t + c2 * t * t) / (
        1 + d1 * t + d2 * t * t + d3 * t * t * t
    )
    return sign * result


def effect_size_cohens_g(b: int, c: int) -> float:
    """
    Cohen's g effect size for McNemar's test.

    g = |p - 0.5| where p = c / (b + c)
    Small: 0.05, Medium: 0.15, Large: 0.25
    """
    n = b + c
    if n == 0:
        return 0.0
    p = c / n
    return abs(p - 0.5)
