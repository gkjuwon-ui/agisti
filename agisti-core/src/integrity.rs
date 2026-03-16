//! Integrity hashing — multi-threaded SHA-256 and fast fingerprinting.
//!
//! The Python version does:
//!   param.cpu().numpy().tobytes() → hashlib.sha256()
//! for EVERY frozen parameter. For 72B models this is catastrophically slow.
//!
//! Rust version:
//! - `parallel_sha256`: rayon + ring for multi-threaded hashing
//! - `fast_fingerprint`: norm+mean+first_element (stays on CPU numpy, no full byte copy)

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;
use ring::digest;

/// SHA-256 of a flat byte buffer (1D numpy array of u8).
#[pyfunction]
pub fn sha256_bytes(data: PyReadonlyArray1<u8>) -> String {
    let slice = data.as_slice().expect("contiguous array required");
    let digest = digest::digest(&digest::SHA256, slice);
    hex_encode(digest.as_ref())
}

/// Compute SHA-256 for multiple byte buffers in parallel.
///
/// `buffers`: list of 1D numpy arrays (each is one parameter's bytes).
/// Returns: list of hex digest strings.
///
/// Uses rayon to hash all parameters concurrently — no GIL.
#[pyfunction]
pub fn parallel_sha256(
    py: Python<'_>,
    buffers: Vec<PyReadonlyArray1<u8>>,
) -> PyResult<Vec<String>> {
    // Copy data out of Python objects so we can release the GIL
    let owned: Vec<Vec<u8>> = buffers
        .iter()
        .map(|b| b.as_slice().expect("contiguous").to_vec())
        .collect();

    let results = py.allow_threads(|| {
        owned
            .par_iter()
            .map(|buf| {
                let d = digest::digest(&digest::SHA256, buf);
                hex_encode(d.as_ref())
            })
            .collect::<Vec<_>>()
    });

    Ok(results)
}

/// Fast fingerprint: compute norm + mean + first element from a flat f32 array.
/// Much faster than SHA-256 because it doesn't need to serialise the entire tensor.
#[pyfunction]
pub fn fast_fingerprint(data: PyReadonlyArray1<f32>) -> String {
    let slice = data.as_slice().expect("contiguous array required");
    if slice.is_empty() {
        return "empty".to_string();
    }

    let mut sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    for &v in slice {
        let vf = v as f64;
        sum += vf;
        sq_sum += vf * vf;
    }
    let n = slice.len() as f64;
    let mean = sum / n;
    let norm = sq_sum.sqrt();
    let first = slice[0] as f64;

    format!("{norm:.10e}|{mean:.10e}|{first:.10e}|{}", slice.len())
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}
