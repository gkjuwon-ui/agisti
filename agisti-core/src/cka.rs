//! CKA (Centered Kernel Alignment) — high-performance computation.
//!
//! The main bottleneck: all-pairs CKA across L_target × L_ref layers.
//! Python does this with nested for-loops (GIL-bound).
//! Rust does it with rayon::par_iter (no GIL, true parallelism).

use ndarray::{Array2, ArrayView2, Axis};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Compute CKA between two activation matrices (debiased).
/// X: (n_samples, dim_x), Y: (n_samples, dim_y)
/// Returns CKA similarity in [0, 1].
#[pyfunction]
pub fn compute_cka_pair(
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray2<f32>,
    debiased: Option<bool>,
) -> PyResult<f64> {
    let x = x.as_array();
    let y = y.as_array();
    let debiased = debiased.unwrap_or(true);

    Ok(cka_impl(x, y, debiased))
}

/// Compute CKA for ALL pairs between two sets of activations.
///
/// `target_acts`: Vec of (n, dim_i) arrays (one per target layer)
/// `ref_acts`:    Vec of (n, dim_j) arrays (one per ref layer)
///
/// Returns: 2D array of shape (n_target, n_ref) with CKA scores.
/// This is the main performance win — rayon parallelises the L×L loop.
#[pyfunction]
pub fn compute_cka_all_pairs<'py>(
    py: Python<'py>,
    target_acts: Vec<PyReadonlyArray2<'py, f32>>,
    ref_acts: Vec<PyReadonlyArray2<'py, f32>>,
    debiased: Option<bool>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let debiased = debiased.unwrap_or(true);
    let n_target = target_acts.len();
    let n_ref = ref_acts.len();

    // Convert to owned ndarray for Send safety
    let t_owned: Vec<Array2<f32>> = target_acts
        .iter()
        .map(|a| a.as_array().to_owned())
        .collect();
    let r_owned: Vec<Array2<f32>> = ref_acts
        .iter()
        .map(|a| a.as_array().to_owned())
        .collect();

    // Release the GIL for the heavy computation
    let results: Vec<f64> = py.allow_threads(|| {
        (0..n_target * n_ref)
            .into_par_iter()
            .map(|idx| {
                let ti = idx / n_ref;
                let ri = idx % n_ref;
                cka_impl(t_owned[ti].view(), r_owned[ri].view(), debiased)
            })
            .collect()
    });

    // Reshape into 2D
    let arr = Array2::from_shape_vec((n_target, n_ref), results)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(PyArray2::from_owned_array(py, arr))
}

/// Debiased HSIC between two kernel matrices (exposed for testing).
#[pyfunction]
pub fn debiased_hsic(
    k: PyReadonlyArray2<f32>,
    l: PyReadonlyArray2<f32>,
) -> PyResult<f64> {
    let k = k.as_array();
    let l = l.as_array();
    Ok(debiased_hsic_impl(k, l))
}

// ─── Internal implementations ──────────────────────────

fn cka_impl(x: ArrayView2<f32>, y: ArrayView2<f32>, debiased: bool) -> f64 {
    let n = x.nrows();
    if n < 4 {
        return 0.0;
    }

    // Linear kernels: K = X @ X^T, L = Y @ Y^T
    let k = matmul_aat(x);
    let l = matmul_aat(y);

    if debiased {
        let hsic_kl = debiased_hsic_impl(k.view(), l.view());
        let hsic_kk = debiased_hsic_impl(k.view(), k.view());
        let hsic_ll = debiased_hsic_impl(l.view(), l.view());

        let product = hsic_kk * hsic_ll;
        if product <= 0.0 {
            return 0.0;
        }
        let denom = product.sqrt();
        if denom < 1e-10 {
            return 0.0;
        }
        (hsic_kl / denom).clamp(0.0, 1.0)
    } else {
        standard_cka_from_kernels(k.view(), l.view(), n)
    }
}

/// Compute A @ A^T for an (n, d) matrix.
fn matmul_aat(a: ArrayView2<f32>) -> Array2<f32> {
    let n = a.nrows();
    let d = a.ncols();
    let mut result = Array2::<f32>::zeros((n, n));

    for i in 0..n {
        for j in i..n {
            let mut sum = 0.0f32;
            for k in 0..d {
                sum += a[[i, k]] * a[[j, k]];
            }
            result[[i, j]] = sum;
            result[[j, i]] = sum; // symmetric
        }
    }
    result
}

fn debiased_hsic_impl(k: ArrayView2<f32>, l: ArrayView2<f32>) -> f64 {
    let n = k.nrows();
    if n < 4 {
        return 0.0;
    }
    let nf = n as f64;

    // Zero-diagonal copies
    let mut k_sum_row = vec![0.0f64; n];
    let mut l_sum_row = vec![0.0f64; n];
    let mut term1 = 0.0f64;
    let mut k_total = 0.0f64;
    let mut l_total = 0.0f64;
    let mut cross = 0.0f64;

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let kv = k[[i, j]] as f64;
            let lv = l[[i, j]] as f64;
            term1 += kv * lv;
            k_sum_row[i] += kv;
            l_sum_row[i] += lv;
            k_total += kv;
            l_total += lv;
        }
    }

    // term2 = k_total * l_total / ((n-1)*(n-2))
    let term2 = k_total * l_total / ((nf - 1.0) * (nf - 2.0));

    // term3 = 2 * sum_i(k_sum_row[i] * l_sum_row[i]) / (n-2)
    for i in 0..n {
        cross += k_sum_row[i] * l_sum_row[i];
    }
    let term3 = 2.0 * cross / (nf - 2.0);

    (term1 + term2 - term3) / (nf * (nf - 3.0))
}

fn standard_cka_from_kernels(k: ArrayView2<f32>, l: ArrayView2<f32>, n: usize) -> f64 {
    let nf = n as f64;
    // Center: H = I - 1/n
    // K_c = H @ K @ H  (simplified: K_c[i,j] = K[i,j] - mean_row_i - mean_row_j + mean_all)
    let k_row_means: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| k[[i, j]] as f64).sum::<f64>() / nf)
        .collect();
    let k_mean: f64 = k_row_means.iter().sum::<f64>() / nf;

    let l_row_means: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| l[[i, j]] as f64).sum::<f64>() / nf)
        .collect();
    let l_mean: f64 = l_row_means.iter().sum::<f64>() / nf;

    let mut hsic_xy = 0.0f64;
    let mut hsic_xx = 0.0f64;
    let mut hsic_yy = 0.0f64;

    for i in 0..n {
        for j in 0..n {
            let kc = k[[i, j]] as f64 - k_row_means[i] - k_row_means[j] + k_mean;
            let lc = l[[i, j]] as f64 - l_row_means[i] - l_row_means[j] + l_mean;
            hsic_xy += kc * lc;
            hsic_xx += kc * kc;
            hsic_yy += lc * lc;
        }
    }

    let denom_sq = hsic_xx * hsic_yy;
    if denom_sq <= 0.0 {
        return 0.0;
    }
    let denom = denom_sq.sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    (hsic_xy / denom).clamp(0.0, 1.0)
}
