//! `tt-core` — Tensor-Train (TT) decomposition library, pure-Rust CPU edition.
//!
//! This is the substrate for `tt-cuda` (GPU kernels) and `tt-py` (PyO3
//! bindings). All operations are exact; numerical tolerance is configurable
//! per call via `(max_rank, tol)`.
//!
//! See [`TtTensor`] for the primary type.

pub mod error;
pub mod tensor;

pub use error::TtError;
pub use tensor::TtTensor;
