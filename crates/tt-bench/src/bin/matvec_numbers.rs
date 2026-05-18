//! Real-numbers comparison: dense vs CPU TT vs GPU TT matvec.
//!
//! Builds low-rank matrices of growing size, decomposes each into TT form,
//! and times the three matvec paths on the same inputs. Prints a table the
//! README pulls from. Skip the GPU column if no CUDA device is available.

use std::time::Instant;

use ndarray::{Array2, ArrayD, IxDyn};
use tt_core::TtMatrix;
use tt_cuda::{CudaTtMatrix, CudaTtRuntime};

/// Build a deterministic rank-`k` matrix of shape (rows, cols): sum of k
/// outer products of mode-shaped factor vectors. Each row/col index is
/// decomposed across `mode_rows` / `mode_cols` for TT factorisation.
fn build_low_rank(rows: usize, cols: usize, rank: usize, seed: u64) -> Array2<f32> {
    let mut a = Array2::<f32>::zeros((rows, cols));
    for k in 0..rank {
        let phase = (seed as f32 + k as f32) * 0.71;
        let u: Vec<f32> = (0..rows).map(|i| ((i as f32) * 0.13 + phase).sin()).collect();
        let v: Vec<f32> = (0..cols).map(|j| ((j as f32) * 0.07 + phase * 0.5).cos()).collect();
        for i in 0..rows {
            for j in 0..cols {
                a[[i, j]] += u[i] * v[j];
            }
        }
    }
    a
}

fn dense_matvec(m: &Array2<f32>, x: &[f32]) -> Vec<f32> {
    let (r, c) = m.dim();
    let mut y = vec![0.0f32; r];
    for i in 0..r {
        let mut s = 0.0f32;
        for j in 0..c {
            s += m[[i, j]] * x[j];
        }
        y[i] = s;
    }
    y
}

fn time<F: FnOnce() -> R, R>(f: F) -> (R, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64() * 1e3)
}

/// Split `n` into `d` factors each ≈ `base`. Assumes n is a power of `base`
/// (the configs in `main` are picked so it is).
fn split_uniform(n: usize, d: usize, base: usize) -> Vec<usize> {
    let v = vec![base; d];
    let prod: usize = v.iter().product();
    debug_assert_eq!(prod, n, "n={n} ≠ base={base} ^ d={d}");
    v
}

fn flatten(shape: &[usize], n: usize, fill: impl Fn(usize) -> f32) -> ArrayD<f32> {
    let data: Vec<f32> = (0..n).map(fill).collect();
    ArrayD::from_shape_vec(IxDyn(shape), data).unwrap()
}

fn main() {
    // Square matrices factored as (mode, …) × (mode, …). Three regimes:
    //   (a) shallow (d=2..3): worst case for TT, best case for dense BLAS
    //   (b) medium  (d=4): the crossover for TT-rank-aware kernels
    //   (c) deep    (d=5..6): where TT really wins on memory and TT-aware
    //                         multiplication algorithms can beat dense.
    // (rows, cols, d, base_mode, true_rank)
    let configs: &[(usize, usize, usize, usize, usize)] = &[
        (16,    16,    2, 4, 4),
        (64,    64,    3, 4, 8),
        (256,   256,   4, 4, 16),
        (1024,  1024,  5, 4, 16),
        (4096,  4096,  6, 4, 16),
        (16384, 16384, 7, 4, 16),
    ];

    let rt = match CudaTtRuntime::new(0) {
        Ok(r) => {
            eprintln!("[gpu] {}", r.device_name());
            Some(r)
        }
        Err(e) => {
            eprintln!("[gpu] not available ({e}); will skip GPU column");
            None
        }
    };
    eprintln!();

    println!("{:>6} {:>6} {:>3} {:>5} {:>14} {:>10} {:>10} {:>11} {:>11}",
        "rows", "cols", "d", "rank", "compression",
        "dense_ms", "cpu_tt_ms", "gpu_naive_ms", "gpu_smart_ms");
    println!("{}", "-".repeat(102));

    for &(rows, cols, d, base, true_rank) in configs {
        if rows.pow(1) != base.pow(d as u32) {
            eprintln!("skipping {}×{}: base^d = {}^{} = {} ≠ rows", rows, cols, base, d, base.pow(d as u32));
            continue;
        }
        let m_shape = split_uniform(rows, d, base);
        let n_shape = split_uniform(cols, d, base);
        let dense = build_low_rank(rows, cols, true_rank, 42);

        // Decompose.
        let (tm, _decomp_ms) = time(|| {
            TtMatrix::from_dense(&m_shape, &n_shape, dense.view(), 32, 1e-6).unwrap()
        });
        let ratio = tm.compression_ratio();
        let ranks: Vec<usize> = tm.tt.ranks();

        // Inputs.
        let x_vec: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.317).sin() + 0.05).collect();
        let x_nd = flatten(&n_shape, cols, |i| x_vec[i]);

        // Time dense matvec.
        let (_y_dense, dense_ms) = time(|| dense_matvec(&dense, &x_vec));

        // Time CPU TT matvec.
        let (_y_cpu, cpu_tt_ms) = time(|| tm.matvec(x_nd.view()).unwrap());

        // Time GPU TT matvec — both kernels, if GPU is available.
        let (gpu_naive_str, gpu_smart_str) = if let Some(rt_arc) = &rt {
            match CudaTtMatrix::upload(rt_arc.clone(), &tm) {
                Ok(cm) => {
                    // warm-up each path
                    let _ = cm.matvec_host(x_nd.view()).unwrap();
                    let _ = cm.matvec_smart_host(x_nd.view()).unwrap();
                    let (_, n_ms) = time(|| cm.matvec_host(x_nd.view()).unwrap());
                    let (_, s_ms) = time(|| cm.matvec_smart_host(x_nd.view()).unwrap());
                    (format!("{n_ms:.3}"), format!("{s_ms:.3}"))
                }
                Err(e) => (format!("err: {e}"), "-".to_string()),
            }
        } else { ("-".to_string(), "-".to_string()) };

        let _ = ranks;

        println!(
            "{:>6} {:>6} {:>3} {:>5} {:>13.1}× {:>10.3} {:>10.3} {:>11} {:>11}",
            rows, cols, d, true_rank, ratio,
            dense_ms, cpu_tt_ms, gpu_naive_str, gpu_smart_str,
        );
    }
    eprintln!();
    eprintln!("note: gpu_tt_ms includes per-call host↔device transfer of x and y,");
    eprintln!("      since the v0.2 API is host-in / host-out. A v0.3 device-resident");
    eprintln!("      API will land before the FlashAttention comparison.");
}
