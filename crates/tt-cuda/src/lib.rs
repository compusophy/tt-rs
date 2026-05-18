//! `tt-cuda` — GPU TT operations.
//!
//! v0.2 ships:
//!   * `smoke`  — end-to-end NVRTC + vector_add toolchain validation.
//!   * `tensor` — `CudaTtRuntime` + `CudaTtTensor` with an on-device
//!                `tt_eval` kernel (chain-of-cores contraction).
//!
//! Coming: `CudaTtMatrix::matvec` (the headline op) and
//!         `tt_attention_fwd` (the v0.4 ambition).

#[cfg(feature = "cuda-11070")]
pub mod smoke;

#[cfg(feature = "cuda-11070")]
pub mod tensor;

#[cfg(feature = "cuda-11070")]
pub mod matrix;

#[cfg(feature = "cuda-11070")]
pub use tensor::{CudaTtRuntime, CudaTtTensor, MAX_RANK};

#[cfg(feature = "cuda-11070")]
pub use matrix::{CudaTtMatrix, MAX_D};
