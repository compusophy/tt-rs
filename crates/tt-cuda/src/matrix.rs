//! GPU TT-matvec — first cut.
//!
//! `CudaTtMatrix` wraps a `tt_core::TtMatrix` whose 2d interleaved cores
//! live on the device. The `tt_matvec` kernel computes y = M x by giving
//! one thread to each output element. The thread iterates over every input
//! index J, contracts the TT-core chain at (I, J) to get the scalar
//! T[I, J], and accumulates T[I,J] · x[J].
//!
//! This is correct but not yet optimal — work scales O(∏m · ∏n · d · r²).
//! A second-generation kernel will share the per-row chain prefix across
//! the J loop and use shared memory; a third-generation kernel will fuse
//! with attention. This one establishes the upload + dispatch shape.

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

use ndarray::{ArrayD, ArrayViewD, IxDyn};
use tt_core::TtMatrix;

use crate::tensor::{CudaTtRuntime, MAX_RANK};

pub const MAX_D: usize = 8;

const KERNEL_SRC: &str = r#"
#define MAX_RANK 64
#define MAX_D 8

extern "C" __global__ void tt_matvec(
    const float* __restrict__ cores,
    const int*   __restrict__ core_offsets,   // length 2d+1
    const int*   __restrict__ core_dims,      // length 3*2d  (r_pre, mode, r_post) per core
    const int*   __restrict__ m_shape,        // length d
    const int*   __restrict__ n_shape,        // length d
    int d,
    const float* __restrict__ x,              // length ∏n
    float*       __restrict__ y,              // length ∏m
    int m_total
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= m_total) return;

    // Decompose out_idx into (i_1, …, i_d) row-major over m_shape.
    int i_idx[MAX_D];
    int rem = out_idx;
    for (int k = d - 1; k >= 0; --k) {
        i_idx[k] = rem % m_shape[k];
        rem /= m_shape[k];
    }

    int n_total = 1;
    for (int k = 0; k < d; ++k) n_total *= n_shape[k];

    float acc = 0.0f;
    int j_idx[MAX_D];

    for (int J = 0; J < n_total; ++J) {
        int rem2 = J;
        for (int k = d - 1; k >= 0; --k) {
            j_idx[k] = rem2 % n_shape[k];
            rem2 /= n_shape[k];
        }

        // Contract the chain. v / w are the running rank-r vector.
        float v[MAX_RANK];
        float w[MAX_RANK];

        // First core: row-core for m_1, shape (1, m_1, r_1).
        int off0 = core_offsets[0];
        int r1   = core_dims[2];
        for (int j = 0; j < r1; ++j) {
            v[j] = cores[off0 + i_idx[0] * r1 + j];
        }
        int r_prev = r1;

        for (int kk = 1; kk < 2 * d; ++kk) {
            int offk = core_offsets[kk];
            int rp   = core_dims[3 * kk + 0];
            int nk   = core_dims[3 * kk + 1];
            int rk   = core_dims[3 * kk + 2];
            int phys = (kk % 2 == 1) ? j_idx[(kk - 1) / 2] : i_idx[kk / 2];

            for (int j = 0; j < rk; ++j) {
                float s = 0.0f;
                for (int p = 0; p < rp; ++p) {
                    s += v[p] * cores[offk + p * nk * rk + phys * rk + j];
                }
                w[j] = s;
            }
            for (int j = 0; j < rk; ++j) v[j] = w[j];
            r_prev = rk;
        }

        acc += v[0] * x[J];
    }

    y[out_idx] = acc;
}
"#;

fn s<E: std::fmt::Debug>(e: E) -> String { format!("{e:?}") }

pub struct CudaTtMatrix {
    runtime: Arc<CudaTtRuntime>,
    cores: CudaSlice<f32>,
    core_offsets: CudaSlice<i32>,
    core_dims: CudaSlice<i32>,
    m_shape_dev: CudaSlice<i32>,
    n_shape_dev: CudaSlice<i32>,
    d: usize,
    m_shape: Vec<usize>,
    n_shape: Vec<usize>,
}

/// Load the `tt_matvec` kernel into the runtime's module store. Idempotent.
pub fn ensure_matvec_loaded(runtime: &CudaTtRuntime) -> Result<(), String> {
    if runtime.device.get_func("tt_matrix_kernels", "tt_matvec").is_some() {
        return Ok(());
    }
    let ptx = compile_ptx(KERNEL_SRC).map_err(s)?;
    runtime.device.load_ptx(ptx, "tt_matrix_kernels", &["tt_matvec"]).map_err(s)?;
    Ok(())
}

impl CudaTtMatrix {
    pub fn upload(runtime: Arc<CudaTtRuntime>, tm: &TtMatrix) -> Result<Self, String> {
        if tm.d() > MAX_D {
            return Err(format!("d={} exceeds MAX_D={MAX_D}", tm.d()));
        }
        if tm.tt.ranks().iter().any(|&r| r > MAX_RANK) {
            return Err(format!(
                "TT-rank exceeds MAX_RANK ({MAX_RANK}); ranks = {:?}",
                tm.tt.ranks()
            ));
        }
        ensure_matvec_loaded(&runtime)?;

        let d = tm.d();
        let n_cores = 2 * d;

        let mut offsets = Vec::with_capacity(n_cores + 1);
        let mut dims = Vec::with_capacity(3 * n_cores);
        let mut flat: Vec<f32> = Vec::new();
        let mut acc: i32 = 0;
        for k in 0..n_cores {
            offsets.push(acc);
            let g = &tm.tt.cores[k];
            let (r_pre, mode, r_post) = (g.shape()[0], g.shape()[1], g.shape()[2]);
            dims.push(r_pre as i32);
            dims.push(mode as i32);
            dims.push(r_post as i32);
            flat.extend(g.iter().copied());
            acc += (r_pre * mode * r_post) as i32;
        }
        offsets.push(acc);

        let m_i32: Vec<i32> = tm.m_shape.iter().map(|&x| x as i32).collect();
        let n_i32: Vec<i32> = tm.n_shape.iter().map(|&x| x as i32).collect();

        let dev = runtime.device.clone();
        let cores = dev.htod_copy(flat).map_err(s)?;
        let core_offsets = dev.htod_copy(offsets).map_err(s)?;
        let core_dims = dev.htod_copy(dims).map_err(s)?;
        let m_shape_dev = dev.htod_copy(m_i32).map_err(s)?;
        let n_shape_dev = dev.htod_copy(n_i32).map_err(s)?;

        Ok(Self {
            runtime,
            cores, core_offsets, core_dims, m_shape_dev, n_shape_dev,
            d,
            m_shape: tm.m_shape.clone(),
            n_shape: tm.n_shape.clone(),
        })
    }

    pub fn rows(&self) -> usize { self.m_shape.iter().product() }
    pub fn cols(&self) -> usize { self.n_shape.iter().product() }

    /// y = M x. Takes a host vector, returns a host vector. (A device-side
    /// matvec API will come once we wire this into an inference loop.)
    pub fn matvec_host(&self, x: ArrayViewD<f32>) -> Result<ArrayD<f32>, String> {
        if x.shape() != self.n_shape.as_slice() {
            return Err(format!(
                "x shape {:?} != n_shape {:?}",
                x.shape(), self.n_shape
            ));
        }
        let dev = &self.runtime.device;
        let x_flat: Vec<f32> = x.iter().copied().collect();
        let x_dev = dev.htod_copy(x_flat).map_err(s)?;
        let m_total = self.rows();
        let mut y_dev = dev.alloc_zeros::<f32>(m_total).map_err(s)?;

        let f = dev
            .get_func("tt_matrix_kernels", "tt_matvec")
            .ok_or_else(|| "kernel tt_matvec not loaded".to_string())?;

        let block: u32 = 128;
        let grid: u32 = ((m_total as u32) + block - 1) / block;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(cfg, (
                &self.cores,
                &self.core_offsets,
                &self.core_dims,
                &self.m_shape_dev,
                &self.n_shape_dev,
                self.d as i32,
                &x_dev,
                &mut y_dev,
                m_total as i32,
            )).map_err(s)?;
        }
        dev.synchronize().map_err(s)?;

        let y_flat = dev.dtoh_sync_copy(&y_dev).map_err(s)?;
        ArrayD::from_shape_vec(IxDyn(&self.m_shape), y_flat).map_err(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn dense_matvec(m: &Array2<f32>, x: &[f32]) -> Vec<f32> {
        let (r, c) = m.dim();
        let mut y = vec![0.0f32; r];
        for i in 0..r {
            let mut s = 0.0f32;
            for j in 0..c { s += m[[i, j]] * x[j]; }
            y[i] = s;
        }
        y
    }

    #[test]
    #[ignore = "requires CUDA-capable GPU; run with `cargo test -- --ignored`"]
    fn cuda_matvec_matches_dense_random() {
        // 16x16 matrix, factored as (4,4) row × (4,4) col.
        let rows = 16usize;
        let cols = 16usize;
        let data_v: Vec<f32> = (0..rows*cols)
            .map(|i| ((i as f32) * 0.31).sin() * 0.7 + ((i as f32) * 0.13).cos() * 0.4)
            .collect();
        let data = Array2::from_shape_vec((rows, cols), data_v.clone()).unwrap();

        let tm = TtMatrix::from_dense(&[4, 4], &[4, 4], data.view(), 8, 1e-8).unwrap();
        let rt = CudaTtRuntime::new(0).expect("init CUDA runtime");
        let cm = CudaTtMatrix::upload(rt, &tm).expect("upload");
        eprintln!("[tt_matvec] ranks = {:?}", tm.tt.ranks());

        let x_v: Vec<f32> = (0..cols).map(|i| ((i as f32) - 7.0) * 0.25).collect();
        let x_arr = ndarray::ArrayD::from_shape_vec(IxDyn(&[4, 4]), x_v.clone()).unwrap();
        let y_gpu: Vec<f32> = cm.matvec_host(x_arr.view()).unwrap().iter().copied().collect();
        let y_dense = dense_matvec(&data, &x_v);
        for (a, b) in y_gpu.iter().zip(y_dense.iter()) {
            assert!((a - b).abs() < 1e-3,
                "matvec mismatch: gpu={} dense={} diff={}", a, b, (a-b).abs());
        }
    }

    #[test]
    #[ignore = "requires CUDA-capable GPU; run with `cargo test -- --ignored`"]
    fn cuda_matvec_identity() {
        let n = 16usize;
        let mut data = Array2::<f32>::zeros((n, n));
        for i in 0..n { data[[i, i]] = 1.0; }
        let tm = TtMatrix::from_dense(&[2,2,2,2], &[2,2,2,2], data.view(), 16, 1e-10).unwrap();
        let rt = CudaTtRuntime::new(0).expect("init CUDA runtime");
        let cm = CudaTtMatrix::upload(rt, &tm).expect("upload");

        let x_v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 1.0).collect();
        let x_arr = ndarray::ArrayD::from_shape_vec(IxDyn(&[2,2,2,2]), x_v.clone()).unwrap();
        let y: Vec<f32> = cm.matvec_host(x_arr.view()).unwrap().iter().copied().collect();
        for (a, b) in x_v.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-3, "identity matvec mismatch: {} vs {}", a, b);
        }
    }
}
