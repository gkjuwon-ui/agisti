//! agisti-core — high–performance Rust backend for AGISTI.
//!
//! Exposes Python bindings via PyO3 for:
//! - CKA (Centered Kernel Alignment) all-pairs computation
//! - Multi-threaded SHA-256 integrity checksumming
//! - Parallel SVD loop helpers (norm-based LoRA budget)
//! - McNemar/binomial statistics (replaces hand-rolled Python math)
//! - Procrustes alignment helpers

use pyo3::prelude::*;

mod cka;
mod integrity;
mod stats;
mod lora;

/// Python module exposed as `import agisti_core`.
#[pymodule]
fn agisti_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // CKA
    m.add_function(wrap_pyfunction!(cka::compute_cka_pair, m)?)?;
    m.add_function(wrap_pyfunction!(cka::compute_cka_all_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(cka::debiased_hsic, m)?)?;

    // Integrity hashing
    m.add_function(wrap_pyfunction!(integrity::sha256_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(integrity::parallel_sha256, m)?)?;
    m.add_function(wrap_pyfunction!(integrity::fast_fingerprint, m)?)?;

    // Statistics
    m.add_function(wrap_pyfunction!(stats::binomial_pmf, m)?)?;
    m.add_function(wrap_pyfunction!(stats::chi2_survival, m)?)?;
    m.add_function(wrap_pyfunction!(stats::comb, m)?)?;
    m.add_function(wrap_pyfunction!(stats::welch_t_test, m)?)?;
    m.add_function(wrap_pyfunction!(stats::wilson_ci, m)?)?;

    // LoRA helpers
    m.add_function(wrap_pyfunction!(lora::parallel_norms, m)?)?;
    m.add_function(wrap_pyfunction!(lora::budget_check, m)?)?;

    Ok(())
}
