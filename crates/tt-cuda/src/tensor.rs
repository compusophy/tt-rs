//! First real TT kernel: element-evaluation on the GPU.
//!
//! `CudaTtRuntime` owns the device and the loaded PTX module. `CudaTtTensor`
//! stores the cores of a `tt_core::TtTensor` on the device as a flattened
//! buffer plus a small metadata table (offsets + per-core (r_pre, n_k, r_k)).
//!
//! `eval_batch(indices)` launches one thread per query. Each thread walks
//! the d cores via a tiny local-memory ping-pong vector (length ≤ MAX_RANK),
//! mirroring the CPU `TtTensor::eval` implementation.
//!
//! This isn't the headline kernel yet (TT-matvec is). It's the substrate
//! that proves we can move TT structures through cudarc and dispatch a
//! contraction kernel correctly on the RTX 3090. The TT-matvec kernel will
//! reuse the same upload path.

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

use tt_core::TtTensor;

/// Compile-time ceiling for the per-thread ping-pong vector. Cores with
/// rank greater than this will be rejected at upload — bumping this just
/// needs a recompile.
pub const MAX_RANK: usize = 64;

const KERNEL_SRC: &str = r#"
#define MAX_RANK 64

extern "C" __global__ void tt_eval(
    const float* __restrict__ cores,
    const int*   __restrict__ core_offsets,   // length d+1, in floats
    const int*   __restrict__ core_dims,      // length 3d:  (r_pre, n_k, r_k) per core
    const int*   __restrict__ indices,        // n_queries * d
    int n_queries,
    int d,
    float* __restrict__ output
) {
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= n_queries) return;

    float v[MAX_RANK];
    float w[MAX_RANK];

    // First core: shape (1, n_0, r_1).
    int off0 = core_offsets[0];
    int n0   = core_dims[1];
    int r1   = core_dims[2];
    int i0   = indices[qi * d + 0];
    // v[j] = cores[off0 + 0 * n0*r1 + i0 * r1 + j]
    #pragma unroll 4
    for (int j = 0; j < r1; ++j) {
        v[j] = cores[off0 + i0 * r1 + j];
    }
    int r_prev = r1;

    for (int k = 1; k < d; ++k) {
        int offk = core_offsets[k];
        int rp = core_dims[3*k + 0];
        int nk = core_dims[3*k + 1];
        int rk = core_dims[3*k + 2];
        int ik = indices[qi * d + k];

        // w[j] = sum_p v[p] * cores[offk + p * nk*rk + ik * rk + j]
        for (int j = 0; j < rk; ++j) {
            float s = 0.0f;
            for (int p = 0; p < rp; ++p) {
                s += v[p] * cores[offk + p * nk * rk + ik * rk + j];
            }
            w[j] = s;
        }
        for (int j = 0; j < rk; ++j) v[j] = w[j];
        r_prev = rk;
    }

    output[qi] = v[0];
}
"#;

/// Owns the device and the loaded PTX module.
pub struct CudaTtRuntime {
    pub device: Arc<CudaDevice>,
}

fn s<E: std::fmt::Debug>(e: E) -> String { format!("{e:?}") }

impl CudaTtRuntime {
    pub fn new(device_idx: usize) -> Result<Arc<Self>, String> {
        let device = CudaDevice::new(device_idx).map_err(s)?;
        let ptx = compile_ptx(KERNEL_SRC).map_err(s)?;
        device.load_ptx(ptx, "tt_kernels", &["tt_eval"]).map_err(s)?;
        Ok(Arc::new(Self { device }))
    }

    pub fn device_name(&self) -> String {
        self.device.name().unwrap_or_else(|_| "<unknown>".to_string())
    }
}

/// A TT-tensor whose cores live on the device. Construction copies the cores
/// once; subsequent `eval_batch` calls reuse them.
pub struct CudaTtTensor {
    runtime: Arc<CudaTtRuntime>,
    cores: CudaSlice<f32>,
    core_offsets: CudaSlice<i32>,
    core_dims: CudaSlice<i32>,
    d: usize,
    shape: Vec<usize>,
}

impl CudaTtTensor {
    pub fn upload(runtime: Arc<CudaTtRuntime>, tt: &TtTensor) -> Result<Self, String> {
        let d = tt.ndim();
        if d == 0 { return Err("empty TT tensor".into()); }
        if tt.ranks().iter().any(|&r| r > MAX_RANK) {
            return Err(format!(
                "TT-rank exceeds MAX_RANK ({MAX_RANK}); ranks = {:?}",
                tt.ranks()
            ));
        }

        let mut offsets = Vec::with_capacity(d + 1);
        let mut dims = Vec::with_capacity(3 * d);
        let mut flat: Vec<f32> = Vec::new();
        let mut acc: i32 = 0;
        for k in 0..d {
            offsets.push(acc);
            let g = &tt.cores[k];
            let (r_pre, n_k, r_post) = (g.shape()[0], g.shape()[1], g.shape()[2]);
            dims.push(r_pre as i32);
            dims.push(n_k as i32);
            dims.push(r_post as i32);
            // ndarray Array3 iter is row-major over (r_pre, n_k, r_post) — matches the kernel.
            flat.extend(g.iter().copied());
            acc += (r_pre * n_k * r_post) as i32;
        }
        offsets.push(acc);

        let dev = runtime.device.clone();
        let cores = dev.htod_copy(flat).map_err(s)?;
        let core_offsets = dev.htod_copy(offsets).map_err(s)?;
        let core_dims = dev.htod_copy(dims).map_err(s)?;
        Ok(Self {
            runtime,
            cores,
            core_offsets,
            core_dims,
            d,
            shape: tt.shape().to_vec(),
        })
    }

    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn ndim(&self) -> usize { self.d }

    /// Evaluate the tensor at a batch of indices. `indices` is laid out as
    /// `n_queries * d` row-major: query q's k-th index is `indices[q*d + k]`.
    pub fn eval_batch(&self, indices: &[i32]) -> Result<Vec<f32>, String> {
        if self.d == 0 { return Ok(vec![]); }
        if indices.len() % self.d != 0 {
            return Err(format!(
                "indices length {} not a multiple of d={}",
                indices.len(), self.d
            ));
        }
        let n_queries = indices.len() / self.d;
        if n_queries == 0 { return Ok(vec![]); }

        let dev = &self.runtime.device;
        let indices_dev = dev.htod_copy(indices.to_vec()).map_err(s)?;
        let mut output_dev = dev.alloc_zeros::<f32>(n_queries).map_err(s)?;

        let f = dev
            .get_func("tt_kernels", "tt_eval")
            .ok_or_else(|| "kernel tt_eval not loaded".to_string())?;

        let block: u32 = 256;
        let grid: u32 = ((n_queries as u32) + block - 1) / block;
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
                &indices_dev,
                n_queries as i32,
                self.d as i32,
                &mut output_dev,
            )).map_err(s)?;
        }
        dev.synchronize().map_err(s)?;
        dev.dtoh_sync_copy(&output_dev).map_err(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, IxDyn};

    fn build_tt(shape: &[usize], gen: impl Fn(&[usize]) -> f32) -> TtTensor {
        let n: usize = shape.iter().product();
        let mut buf = Vec::with_capacity(n);
        let mut idx = vec![0usize; shape.len()];
        for flat in 0..n {
            let mut rem = flat;
            for k in (0..shape.len()).rev() {
                idx[k] = rem % shape[k];
                rem /= shape[k];
            }
            buf.push(gen(&idx));
        }
        let arr = Array::from_shape_vec(IxDyn(shape), buf).unwrap();
        TtTensor::from_dense(arr.view(), 32, 1e-10).unwrap()
    }

    #[test]
    #[ignore = "requires CUDA-capable GPU; run with `cargo test -- --ignored`"]
    fn cuda_eval_matches_cpu_random_tensor() {
        let shape = [4, 5, 6, 3];
        let tt = build_tt(&shape, |idx| {
            ((idx[0] as f32 * 0.7
              + idx[1] as f32 * 0.3
              + idx[2] as f32 * 0.5
              + idx[3] as f32 * 0.9)).sin() + 0.2
        });

        let rt = CudaTtRuntime::new(0).expect("init CUDA runtime");
        eprintln!("[tt_eval] device: {}", rt.device_name());
        let dev_tt = CudaTtTensor::upload(rt, &tt).unwrap();

        // Build the full index grid; should match CPU eval bit-close.
        let mut indices: Vec<i32> = Vec::new();
        let mut idx = vec![0usize; shape.len()];
        let total: usize = shape.iter().product();
        for flat in 0..total {
            let mut rem = flat;
            for k in (0..shape.len()).rev() {
                idx[k] = rem % shape[k];
                rem /= shape[k];
            }
            for &i in &idx { indices.push(i as i32); }
        }

        let gpu = dev_tt.eval_batch(&indices).unwrap();
        assert_eq!(gpu.len(), total);

        for q in 0..total {
            let q_idx: Vec<usize> = indices[q * shape.len()..(q + 1) * shape.len()]
                .iter().map(|&i| i as usize).collect();
            let cpu = tt.eval(&q_idx).unwrap();
            assert!(
                (gpu[q] - cpu).abs() < 1e-4,
                "mismatch at query {q} idx={:?}: gpu={} cpu={}",
                q_idx, gpu[q], cpu
            );
        }
    }
}
