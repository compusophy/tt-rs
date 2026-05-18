//! `tt-cuda` — GPU TT operations. v0.1 ships a smoke test that proves the
//! CUDA toolchain works end-to-end on the host. Real TT kernels (TT-matvec,
//! TT-attention) land in v0.2.

#[cfg(feature = "cuda-11070")]
pub mod smoke;
