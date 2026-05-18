//! End-to-end CUDA smoke test: NVRTC-compile a vector-add kernel, launch it
//! on the device, copy back the result, verify. Used to validate the
//! toolchain (cudarc + CUDA 11.7 + RTX 3090 driver) before writing the real
//! TT kernels.

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

const KERNEL: &str = r#"
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
"#;

/// Run the smoke test. Returns Ok(()) iff cudart can be initialized, the
/// kernel compiles via NVRTC, and the device-side result matches CPU.
pub fn run() -> Result<(), String> {
    fn s<E: std::fmt::Debug>(e: E) -> String { format!("{e:?}") }

    // 1) acquire device 0
    let dev: Arc<CudaDevice> = CudaDevice::new(0).map_err(s)?;
    eprintln!("[smoke] device 0 opened: {:?}", dev.name().map_err(s)?);

    // 2) compile kernel via NVRTC
    let ptx = compile_ptx(KERNEL).map_err(s)?;
    dev.load_ptx(ptx, "smoke", &["vector_add"]).map_err(s)?;
    let f = dev.get_func("smoke", "vector_add").ok_or_else(|| "kernel not loaded".to_string())?;

    // 3) host data
    let n: usize = 1 << 16; // 64k floats
    let a_host: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let b_host: Vec<f32> = (0..n).map(|i| (i as f32) * -0.3 + 1.0).collect();

    // 4) ship to device
    let a_dev = dev.htod_copy(a_host.clone()).map_err(s)?;
    let b_dev = dev.htod_copy(b_host.clone()).map_err(s)?;
    let mut c_dev = dev.alloc_zeros::<f32>(n).map_err(s)?;

    // 5) launch
    let block = 256u32;
    let grid = ((n as u32) + block - 1) / block;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { f.launch(cfg, (&a_dev, &b_dev, &mut c_dev, n)).map_err(s)?; }
    dev.synchronize().map_err(s)?;

    // 6) copy back & verify
    let c_host: Vec<f32> = dev.dtoh_sync_copy(&c_dev).map_err(s)?;
    for i in 0..n {
        let expected = a_host[i] + b_host[i];
        if (c_host[i] - expected).abs() > 1e-4 {
            return Err(format!("mismatch at {i}: gpu={} cpu={}", c_host[i], expected));
        }
    }
    eprintln!("[smoke] {n} adds OK on {:?}", dev.name().map_err(s)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA-capable GPU; run with `cargo test -- --ignored`"]
    fn vector_add_runs_on_gpu() {
        run().expect("smoke test failed");
    }
}
