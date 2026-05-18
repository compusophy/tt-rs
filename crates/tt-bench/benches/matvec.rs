//! Decomposition and storage benchmarks for `tt-core`.
//!
//! Demonstrates the storage compression win and decomposition throughput on a
//! few representative tensors. Phase v0.2 will add CUDA matvec comparisons.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::{Array, IxDyn};
use tt_core::TtTensor;

fn rank1_data(shape: &[usize]) -> Vec<f32> {
    // Outer product of (1..n_k) vectors — exactly rank-1 in TT form.
    let n: usize = shape.iter().product();
    let factors: Vec<Vec<f32>> = shape.iter()
        .enumerate()
        .map(|(d, &n_d)| (0..n_d).map(|i| ((i + d) as f32).sin() + 0.1).collect())
        .collect();
    let mut data = vec![1.0f32; n];
    let strides = {
        let mut s = vec![0usize; shape.len()];
        let mut acc = 1usize;
        for k in (0..shape.len()).rev() {
            s[k] = acc;
            acc *= shape[k];
        }
        s
    };
    for (flat, slot) in data.iter_mut().enumerate() {
        let mut v = 1.0f32;
        let mut rem = flat;
        for k in 0..shape.len() {
            let idx = rem / strides[k];
            rem %= strides[k];
            v *= factors[k][idx];
        }
        *slot = v;
    }
    data
}

fn full_rank_data(shape: &[usize]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    (0..n).map(|i| ((i as f32) * 1.7).sin() + ((i as f32) * 0.91).cos() * 0.5).collect()
}

fn bench_decompose(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_dense");
    for &shape in &[
        &[8, 8, 8][..],
        &[16, 16, 16][..],
        &[8, 8, 8, 8][..],
    ] {
        let data = full_rank_data(shape);
        let arr = Array::from_shape_vec(IxDyn(shape), data).unwrap();
        let n: usize = shape.iter().product();
        group.bench_with_input(
            BenchmarkId::new("full_rank", format!("{:?} ({} elem)", shape, n)),
            &arr,
            |b, arr| b.iter(|| TtTensor::from_dense(arr.view(), 64, 1e-10).unwrap()),
        );
    }
    group.finish();
}

fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_ratio");
    for &shape in &[
        &[8, 8, 8, 8][..],
        &[16, 16, 16, 16][..],
    ] {
        let data = rank1_data(shape);
        let arr = Array::from_shape_vec(IxDyn(shape), data).unwrap();
        let tt = TtTensor::from_dense(arr.view(), 8, 1e-8).unwrap();
        let n: usize = shape.iter().product();
        println!(
            "shape {:?}: dense = {} f32, tt = {} f32, ratio = {:.1}×, ranks = {:?}",
            shape, n, tt.n_params(), tt.compression_ratio(), tt.ranks()
        );
        group.bench_with_input(
            BenchmarkId::new("eval", format!("{:?}", shape)),
            &tt,
            |b, tt| {
                let idx: Vec<usize> = shape.iter().map(|&n| n / 2).collect();
                b.iter(|| black_box(tt.eval(&idx).unwrap()))
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_decompose, bench_compression);
criterion_main!(benches);
