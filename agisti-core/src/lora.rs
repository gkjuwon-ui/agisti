//! LoRA helpers — parallel norm computation and budget enforcement.

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Compute Frobenius norms of A @ B products in parallel.
///
/// `a_list`: list of 2D arrays (each is a LoRA A matrix)
/// `b_list`: list of 2D arrays (each is a LoRA B matrix)
///
/// Returns: list of Frobenius norms ||A @ B||_F.
/// Uses rayon to compute all norms simultaneously.
#[pyfunction]
pub fn parallel_norms(
    py: Python<'_>,
    a_list: Vec<PyReadonlyArray2<f32>>,
    b_list: Vec<PyReadonlyArray2<f32>>,
) -> PyResult<Vec<f64>> {
    if a_list.len() != b_list.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "a_list and b_list must have same length",
        ));
    }

    // Copy data out of Python objects
    let pairs: Vec<(Vec<f32>, usize, usize, Vec<f32>, usize, usize)> = a_list
        .iter()
        .zip(b_list.iter())
        .map(|(a, b)| {
            let a_arr = a.as_array();
            let b_arr = b.as_array();
            (
                a_arr.iter().cloned().collect(),
                a_arr.nrows(),
                a_arr.ncols(),
                b_arr.iter().cloned().collect(),
                b_arr.nrows(),
                b_arr.ncols(),
            )
        })
        .collect();

    let results = py.allow_threads(|| {
        pairs
            .par_iter()
            .map(|(a_data, a_rows, a_cols, b_data, b_rows, b_cols)| {
                frobenius_norm_product(a_data, *a_rows, *a_cols, b_data, *b_rows, *b_cols)
            })
            .collect::<Vec<_>>()
    });

    Ok(results)
}

/// Check if total delta norm exceeds budget.
///
/// `norms`: list of per-layer norms
/// `budget`: maximum allowed total norm
///
/// Returns: (total_norm, within_budget)
#[pyfunction]
pub fn budget_check(norms: Vec<f64>, budget: f64) -> (f64, bool) {
    let total: f64 = norms.iter().map(|n| n * n).sum::<f64>().sqrt();
    (total, total <= budget)
}

/// Compute ||A @ B||_F without materialising the full product.
/// A: (m, k), B: (k, n) → ||AB||_F = sqrt(sum of squared elements of AB)
fn frobenius_norm_product(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    _bk: usize,
    n: usize,
) -> f64 {
    let mut sq_sum = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0f64;
            for p in 0..k {
                val += a[i * k + p] as f64 * b[p * n + j] as f64;
            }
            sq_sum += val * val;
        }
    }
    sq_sum.sqrt()
}
