//! Minimal single-head attention through TT-compressed K and V.
//!
//! For each query q_i in Q (shape [seq, dim]):
//!   1. scores_i  = K · q_i               via TT-matvec
//!   2. probs_i   = softmax(scores_i / √d)
//!   3. out_i     = V_t · probs_i         via TT-matvec
//!      where V_t is V stored already transposed (shape [dim, seq]).
//!
//! Both K and V_t are compressed into TT-matrices once before the loop;
//! Q stays dense (it varies per call in a real attention layer). The
//! output is compared against the dense reference attention computed
//! the textbook way, and we report compression ratio + max-abs-error.

use std::time::Instant;

use ndarray::{Array2, ArrayD, ArrayView2, IxDyn};
use tt_core::TtMatrix;

fn softmax(scores: &mut [f32], scale: f32) {
    for s in scores.iter_mut() { *s *= scale; }
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() { *s = (*s - max_s).exp(); sum += *s; }
    for s in scores.iter_mut() { *s /= sum; }
}

fn dense_attention(
    q: ArrayView2<f32>,
    k: ArrayView2<f32>,
    v: ArrayView2<f32>,
) -> Array2<f32> {
    let (seq, dim) = q.dim();
    let (kseq, _kdim) = k.dim();
    assert_eq!(seq, kseq);
    let scale = 1.0 / (dim as f32).sqrt();
    let mut out = Array2::<f32>::zeros((seq, dim));
    let mut scratch = vec![0.0f32; seq];

    for i in 0..seq {
        for j in 0..seq {
            let mut s = 0.0f32;
            for d in 0..dim { s += q[[i, d]] * k[[j, d]]; }
            scratch[j] = s;
        }
        softmax(&mut scratch, scale);
        for d in 0..dim {
            let mut s = 0.0f32;
            for j in 0..seq { s += scratch[j] * v[[j, d]]; }
            out[[i, d]] = s;
        }
    }
    out
}

fn tt_attention(
    q: ArrayView2<f32>,
    k_tt: &TtMatrix,                  // shape (seq, dim)
    v_t_tt: &TtMatrix,                // shape (dim, seq) — V transposed
) -> Array2<f32> {
    let (seq, dim) = q.dim();
    let scale = 1.0 / (dim as f32).sqrt();
    let mut out = Array2::<f32>::zeros((seq, dim));

    for i in 0..seq {
        // q_i shape (dim,) — reshape to k_tt's n_shape
        let q_i: Vec<f32> = q.row(i).to_owned().to_vec();
        let q_nd = ArrayD::from_shape_vec(IxDyn(&k_tt.n_shape), q_i).unwrap();
        let scores_arr = k_tt.matvec(q_nd.view()).expect("K matvec");
        let mut scores_flat: Vec<f32> = scores_arr.iter().copied().collect();
        softmax(&mut scores_flat, scale);

        // out_i = V_t · probs_i, shape (dim,)
        let probs_nd = ArrayD::from_shape_vec(&v_t_tt.n_shape[..], scores_flat).unwrap();
        let out_i = v_t_tt.matvec(probs_nd.view()).expect("V_t matvec");
        for (d_idx, &val) in out_i.iter().enumerate() {
            out[[i, d_idx]] = val;
        }
    }
    out
}

fn build_low_effective_rank(rows: usize, cols: usize, eff_rank: usize, phase: f32) -> Array2<f32> {
    // Sum of `eff_rank` outer products → matrix that compresses well into TT.
    let mut m = Array2::<f32>::zeros((rows, cols));
    for k in 0..eff_rank {
        let u: Vec<f32> = (0..rows).map(|i| ((i as f32) * 0.13 + (k as f32) * 1.7 + phase).sin()).collect();
        let v: Vec<f32> = (0..cols).map(|j| ((j as f32) * 0.07 + (k as f32) * 0.9 + phase * 0.5).cos()).collect();
        for i in 0..rows {
            for j in 0..cols {
                m[[i, j]] += u[i] * v[j];
            }
        }
    }
    m
}

fn time_ms<F: FnOnce() -> R, R>(f: F) -> (R, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64() * 1e3)
}

fn main() {
    // Single-head attention shapes. Pick seq and dim to factor cleanly into
    // a multi-mode TT layout.
    let seq = 4096usize;   // 4 * 4 * 4 * 4 * 4 * 4
    let dim = 64usize;     // 4 * 4 * 4
    let seq_factors = [4, 4, 4, 4, 4, 4];
    let dim_factors = [4, 4, 4];
    let eff_rank = 8usize;

    println!("attention-demo: single-head TT-compressed K, V");
    println!("  seq = {seq}   dim = {dim}   effective rank = {eff_rank}");
    println!();

    // Build matrices with low effective rank.
    let q = build_low_effective_rank(seq, dim, eff_rank, 0.0);
    let k = build_low_effective_rank(seq, dim, eff_rank, 1.0);
    let v = build_low_effective_rank(seq, dim, eff_rank, 2.5);

    // K shape (seq, dim) factored as (seq_factors) × (dim_factors).
    // The TT depth is determined by the longer factor list; pad the shorter
    // one with 1s so both have the same length.
    let d = seq_factors.len();
    let mut m_shape: Vec<usize> = seq_factors.to_vec();
    let mut n_shape: Vec<usize> = dim_factors.to_vec();
    while n_shape.len() < d { n_shape.push(1); }
    while m_shape.len() < d { m_shape.push(1); }
    assert_eq!(m_shape.iter().product::<usize>(), seq);
    assert_eq!(n_shape.iter().product::<usize>(), dim);

    let max_rank = 32;
    let tol = 1e-5;

    println!("compressing K shape ({}, {}) as {:?} × {:?} ...", seq, dim, m_shape, n_shape);
    let (k_tt, k_decomp_ms) = time_ms(|| {
        TtMatrix::from_dense(&m_shape, &n_shape, k.view(), max_rank, tol).unwrap()
    });

    // V transposed: shape (dim, seq) factored as (dim_factors padded) × (seq_factors).
    let mut m_shape_t: Vec<usize> = dim_factors.to_vec();
    let mut n_shape_t: Vec<usize> = seq_factors.to_vec();
    while m_shape_t.len() < d { m_shape_t.push(1); }
    while n_shape_t.len() < d { n_shape_t.push(1); }
    let v_t: Array2<f32> = v.t().to_owned();
    println!("compressing V^T shape ({}, {}) as {:?} × {:?} ...", dim, seq, m_shape_t, n_shape_t);
    let (v_t_tt, v_decomp_ms) = time_ms(|| {
        TtMatrix::from_dense(&m_shape_t, &n_shape_t, v_t.view(), max_rank, tol).unwrap()
    });

    println!();
    println!("decomposition:  K {:.1} ms, V {:.1} ms", k_decomp_ms, v_decomp_ms);
    println!("  K: {}", k_tt);
    println!("  V_t: {}", v_t_tt);

    // Run dense vs TT attention.
    println!();
    let (out_dense, dense_ms) = time_ms(|| dense_attention(q.view(), k.view(), v.view()));
    let (out_tt, tt_ms) = time_ms(|| tt_attention(q.view(), &k_tt, &v_t_tt));

    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f64;
    for (a, b) in out_dense.iter().zip(out_tt.iter()) {
        let d = (a - b).abs();
        if d > max_abs { max_abs = d; }
        sum_sq += (d as f64) * (d as f64);
    }
    let rms = (sum_sq / (out_dense.len() as f64)).sqrt() as f32;

    let dense_bytes_kv = 2 * seq * dim * std::mem::size_of::<f32>();
    let tt_bytes_kv = (k_tt.n_params() + v_t_tt.n_params()) * std::mem::size_of::<f32>();
    let kv_compression = dense_bytes_kv as f32 / tt_bytes_kv.max(1) as f32;

    println!();
    println!("KV memory:");
    println!("  dense: {} KB", dense_bytes_kv / 1024);
    println!("     tt: {} KB", tt_bytes_kv / 1024);
    println!("  ratio: {:.1}×", kv_compression);
    println!();
    println!("attention time:");
    println!("  dense: {:.1} ms", dense_ms);
    println!("     tt: {:.1} ms", tt_ms);
    println!();
    println!("output diff vs dense reference:");
    println!("  max-abs: {:.3e}", max_abs);
    println!("  rms    : {:.3e}", rms);
}
