//! `tt-cuda` — GPU TT operations. Stub for v0.1; kernels land in v0.2.
//!
//! Planned exports:
//! - `CudaTtTensor::upload(&TtTensor, &CudaDevice)` — push cores to device fp16
//! - `CudaTtTensor::matvec(x, y)` — fused TT-matvec compute kernel
//! - `tt_attention_fwd(q, k_tt, v_tt, out, mask)` — fused TT-attention
