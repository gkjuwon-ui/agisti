//! Statistical functions — replaces hand-rolled Python math.
//!
//! Provides: binomial PMF, chi-squared survival, combinations,
//! Welch's t-test, Wilson CI.  All pure Rust, no scipy dependency.

use pyo3::prelude::*;

/// Binomial coefficient C(n, k) using multiplicative formula.
#[pyfunction]
pub fn comb(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result: u64 = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Binomial PMF: P(X = k) where X ~ Binomial(n, p).
#[pyfunction]
pub fn binomial_pmf(n: u64, k: u64, p: f64) -> f64 {
    if k > n {
        return 0.0;
    }
    let coeff = comb(n, k) as f64;
    coeff * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
}

/// Survival function (1 - CDF) of chi-squared distribution.
/// For df=1: P(X > x) = erfc(sqrt(x/2)).
#[pyfunction]
pub fn chi2_survival(x: f64, df: Option<u32>) -> f64 {
    let df = df.unwrap_or(1);
    if x <= 0.0 {
        return 1.0;
    }
    if df == 1 {
        return erfc((x / 2.0).sqrt());
    }
    regularized_upper_gamma(df as f64 / 2.0, x / 2.0)
}

/// Welch's t-test for two samples.
/// Returns (t_statistic, p_value).
#[pyfunction]
pub fn welch_t_test(a: Vec<f64>, b: Vec<f64>) -> (f64, f64) {
    let n_a = a.len();
    let n_b = b.len();
    if n_a < 2 || n_b < 2 {
        return (0.0, 1.0);
    }

    let mean_a: f64 = a.iter().sum::<f64>() / n_a as f64;
    let mean_b: f64 = b.iter().sum::<f64>() / n_b as f64;
    let var_a: f64 = a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1) as f64;
    let var_b: f64 = b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1) as f64;

    let se = (var_a / n_a as f64 + var_b / n_b as f64).sqrt();
    if se < 1e-12 {
        return (0.0, 1.0);
    }

    let t_stat = (mean_a - mean_b) / se;
    let p_value = 2.0 * normal_cdf(-t_stat.abs());
    (t_stat, p_value)
}

/// Wilson score confidence interval for a proportion.
/// Returns (lower, upper).
#[pyfunction]
pub fn wilson_ci(p_hat: f64, n: u64, alpha: Option<f64>) -> (f64, f64) {
    let alpha = alpha.unwrap_or(0.05);
    if n == 0 {
        return (0.0, 1.0);
    }
    let nf = n as f64;
    let z = normal_ppf(1.0 - alpha / 2.0);
    let z2 = z * z;

    let denom = 1.0 + z2 / nf;
    let center = (p_hat + z2 / (2.0 * nf)) / denom;
    let spread =
        z * ((p_hat * (1.0 - p_hat) + z2 / (4.0 * nf)) / nf).sqrt() / denom;

    ((center - spread).max(0.0), (center + spread).min(1.0))
}

// ─── Internal helpers ────────────────────────────────

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn normal_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    let (sign, p_adj) = if p < 0.5 { (-1.0, 1.0 - p) } else { (1.0, p) };
    let t = (-2.0 * (1.0 - p_adj).ln()).sqrt();

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    sign * (t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t))
}

/// Error function approximation (Abramowitz & Stegun 7.1.26).
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Complementary error function.
fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

/// Upper regularised incomplete gamma Q(a, x).
fn regularized_upper_gamma(a: f64, x: f64) -> f64 {
    if x < a + 1.0 {
        1.0 - lower_gamma_series(a, x)
    } else {
        upper_gamma_cf(a, x)
    }
}

fn lower_gamma_series(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    let mut term = 1.0 / a;
    let mut total = term;
    for n in 1..200 {
        term *= x / (a + n as f64);
        total += term;
        if term.abs() < 1e-15 * total.abs() {
            break;
        }
    }
    total * (-x + a * x.ln() - lgamma(a)).exp()
}

fn upper_gamma_cf(a: f64, x: f64) -> f64 {
    let mut d = 1.0 / (x + 1.0 - a);
    let mut h = d;

    for i in 1..200 {
        let an = -(i as f64) * (i as f64 - a);
        let bn = x + 2.0 * i as f64 + 1.0 - a;
        d = an * d + bn;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        let mut c = bn + an / 1e-30_f64.max(bn);
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = c * d;
        h *= delta;
        if (delta - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x + a * x.ln() - lgamma(a)).exp() * h
}

/// Log gamma via Stirling's approximation.
fn lgamma(x: f64) -> f64 {
    // Use Lanczos approximation (g=7)
    let coeffs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - lgamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut a = coeffs[0];
    let t = x + 7.5;
    for (i, &c) in coeffs[1..].iter().enumerate() {
        a += c / (x + i as f64 + 1.0);
    }

    0.5 * (2.0 * std::f64::consts::PI).ln() + (t).ln() * (x + 0.5) - t + a.ln()
}
