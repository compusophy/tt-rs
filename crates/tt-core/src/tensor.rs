//! Tensor-Train (TT) format for a d-dimensional tensor of shape (n_1, ..., n_d).
//!
//! Cores: `Vec<Array3<f32>>` of length d, where `cores[k]` has shape
//! `(r_{k-1}, n_k, r_k)` with `r_0 = r_d = 1`.
//!
//! Element evaluation:
//!     T[i_1, ..., i_d] = G_1[0, i_1, :] · G_2[:, i_2, :] · ... · G_d[:, i_d, 0]
//! a chain of (row-vector × matrix) products.
//!
//! Storage: O(d · n_max · r_max²) instead of O(n^d). The compression win is
//! the small effective TT ranks `r_k`. For random tensors, ranks saturate
//! (r_k = min(prod n_1..k, prod n_{k+1..d})); for low-rank-structured tensors
//! they stay small and TT crushes dense storage.

use ndarray::{Array3, ArrayD, ArrayViewD, IxDyn};

use crate::error::TtError;

#[derive(Debug, Clone)]
pub struct TtTensor {
    /// d cores. `cores[k].shape() == (r_{k-1}, n_k, r_k)` with r_0 = r_d = 1.
    pub cores: Vec<Array3<f32>>,
    /// Mode sizes (n_1, n_2, ..., n_d).
    pub shape: Vec<usize>,
}

impl TtTensor {
    pub fn ndim(&self) -> usize { self.cores.len() }

    pub fn shape(&self) -> &[usize] { &self.shape }

    /// (r_0, r_1, ..., r_d). r_0 = r_d = 1.
    pub fn ranks(&self) -> Vec<usize> {
        let mut r = vec![1usize];
        for c in &self.cores { r.push(c.shape()[2]); }
        r
    }

    /// Number of stored f32 parameters.
    pub fn n_params(&self) -> usize {
        self.cores.iter().map(|c| c.len()).sum()
    }

    /// Number of f32 the equivalent dense tensor would store.
    pub fn dense_n_params(&self) -> usize {
        self.shape.iter().product()
    }

    /// dense / tt. >1 means TT is smaller.
    pub fn compression_ratio(&self) -> f32 {
        self.dense_n_params() as f32 / self.n_params().max(1) as f32
    }

    /// TT-SVD decomposition.
    /// - `max_rank` caps every TT-rank.
    /// - `tol` drops singular values below `tol * σ_max` at each unfolding.
    ///
    /// Complexity: O(d · n^{d+1}) for the full set of unfoldings — fine for
    /// small d and a starter library, expensive for big tensors. v0.2 will
    /// add a randomized variant.
    pub fn from_dense(
        t: ArrayViewD<f32>,
        max_rank: usize,
        tol: f32,
    ) -> Result<Self, TtError> {
        if max_rank == 0 { return Err(TtError::InvalidRank { got: 0 }); }
        let shape: Vec<usize> = t.shape().to_vec();
        let d = shape.len();
        if d == 0 { return Err(TtError::EmptyTensor); }
        let n_total: usize = shape.iter().product();

        // Walking buffer for the un-decomposed tail, in f64 for SVD precision.
        let mut c: Vec<f64> = t.iter().map(|&v| v as f64).collect();
        debug_assert_eq!(c.len(), n_total);

        let mut cores: Vec<Array3<f32>> = Vec::with_capacity(d);
        let mut r_prev = 1usize;
        let mut tail = n_total;

        for k in 0..d {
            let n_k = shape[k];
            tail /= n_k;
            let left = r_prev * n_k;
            let right = tail;

            if k == d - 1 {
                debug_assert_eq!(c.len(), r_prev * n_k);
                let core_f32: Vec<f32> = c.iter().map(|&v| v as f32).collect();
                let core = Array3::from_shape_vec((r_prev, n_k, 1), core_f32)
                    .map_err(|_| TtError::SvdFailed)?;
                cores.push(core);
                break;
            }

            // Reshape c into a (left, right) f64 matrix and SVD.
            let mat: nalgebra::DMatrix<f64> =
                nalgebra::DMatrix::from_fn(left, right, |i, j| c[i * right + j]);
            let svd = mat.svd(true, true);
            let u  = svd.u.ok_or(TtError::SvdFailed)?;
            let s  = svd.singular_values;
            let vt = svd.v_t.ok_or(TtError::SvdFailed)?;

            let n_sv = s.len();
            let sigma_max = s.iter().fold(0.0f64, |a, &b| a.max(b));
            let cutoff = (tol as f64) * sigma_max;
            let mut r_k = s.iter().filter(|&&v| v > cutoff).count();
            r_k = r_k.min(max_rank).min(n_sv).max(1);

            // New core = first r_k columns of U as f32, reshaped to (r_prev, n_k, r_k).
            let mut core = Array3::<f32>::zeros((r_prev, n_k, r_k));
            for a in 0..r_prev {
                for ni in 0..n_k {
                    let row = a * n_k + ni;
                    for j in 0..r_k {
                        core[[a, ni, j]] = u[(row, j)] as f32;
                    }
                }
            }
            cores.push(core);

            // c <- diag(s[..r_k]) @ vt[..r_k, :]   (row-major (r_k, right)), f64
            let mut new_c = vec![0.0f64; r_k * right];
            for i in 0..r_k {
                let sigma = s[i];
                for j in 0..right {
                    new_c[i * right + j] = sigma * vt[(i, j)];
                }
            }
            c = new_c;
            r_prev = r_k;
        }

        Ok(Self { cores, shape })
    }

    /// Reconstruct the dense tensor from the cores. O(prod(shape) · r_max).
    pub fn to_dense(&self) -> ArrayD<f32> {
        let d = self.cores.len();
        // `acc` holds the partial contraction reshaped as (prefix..., r_k).
        // After processing core 0, acc has shape (n_0, r_1).
        let g0 = &self.cores[0];
        let (a0, n0, b0) = (g0.shape()[0], g0.shape()[1], g0.shape()[2]);
        debug_assert_eq!(a0, 1);
        let mut acc: Vec<f32> = g0.iter().copied().collect(); // (1, n_0, b_0) row-major == (n_0, b_0)
        let mut acc_shape: Vec<usize> = vec![n0, b0];

        for k in 1..d {
            let g = &self.cores[k];
            let (r_prev, n_k, r_k) = (g.shape()[0], g.shape()[1], g.shape()[2]);
            debug_assert_eq!(r_prev, *acc_shape.last().unwrap());
            let prefix: usize = acc_shape[..acc_shape.len() - 1].iter().product();

            // acc:  (prefix, r_prev)
            // g:    (r_prev, n_k, r_k) viewed as (r_prev, n_k * r_k)
            // out:  (prefix, n_k * r_k)
            let mut out = vec![0.0f32; prefix * n_k * r_k];
            for p in 0..prefix {
                for nk in 0..n_k {
                    for rk in 0..r_k {
                        let mut s = 0.0f32;
                        for rp in 0..r_prev {
                            s += acc[p * r_prev + rp] * g[[rp, nk, rk]];
                        }
                        out[p * (n_k * r_k) + nk * r_k + rk] = s;
                    }
                }
            }
            acc = out;
            acc_shape.pop();
            acc_shape.push(n_k);
            acc_shape.push(r_k);
        }
        // Drop trailing r_d = 1.
        let last = acc_shape.pop().unwrap_or(1);
        debug_assert_eq!(last, 1);

        ArrayD::from_shape_vec(IxDyn(&acc_shape), acc)
            .expect("dimension product mismatch in to_dense — bug")
    }

    /// Single element T[i_1, ..., i_d].
    pub fn eval(&self, idx: &[usize]) -> Result<f32, TtError> {
        let d = self.ndim();
        if idx.len() != d {
            return Err(TtError::IndexDimMismatch { ndim: d, got: idx.len() });
        }
        for k in 0..d {
            if idx[k] >= self.shape[k] {
                return Err(TtError::IndexOutOfBounds {
                    dim: k, size: self.shape[k], got: idx[k],
                });
            }
        }
        // v starts as G_1[0, i_1, :] (length r_1).
        let g0 = &self.cores[0];
        let r1 = g0.shape()[2];
        let mut v: Vec<f32> = (0..r1).map(|j| g0[[0, idx[0], j]]).collect();
        for k in 1..d {
            let g = &self.cores[k];
            let (r_prev, _n_k, r_k) = (g.shape()[0], g.shape()[1], g.shape()[2]);
            debug_assert_eq!(r_prev, v.len());
            let mut nv = vec![0.0f32; r_k];
            for j in 0..r_k {
                let mut s = 0.0f32;
                for ip in 0..r_prev {
                    s += v[ip] * g[[ip, idx[k], j]];
                }
                nv[j] = s;
            }
            v = nv;
        }
        debug_assert_eq!(v.len(), 1);
        Ok(v[0])
    }

    /// ‖T‖² computed directly in TT form (no dense reconstruction).
    /// Cost O(d · n_max · r_max³).
    pub fn norm_sq(&self) -> f32 {
        if self.cores.is_empty() { return 0.0; }
        let mut w = vec![1.0f32]; // (1, 1)
        let mut w_dim = 1usize;
        for g in &self.cores {
            let (r_prev, n_k, r_k) = (g.shape()[0], g.shape()[1], g.shape()[2]);
            debug_assert_eq!(r_prev, w_dim);
            let mut w_new = vec![0.0f32; r_k * r_k];
            for i in 0..n_k {
                // tmp[b, a'] = sum_a G[a, i, b] * W[a, a']      shape (r_k, r_prev)
                let mut tmp = vec![0.0f32; r_k * r_prev];
                for b in 0..r_k {
                    for ap in 0..r_prev {
                        let mut s = 0.0f32;
                        for a in 0..r_prev {
                            s += g[[a, i, b]] * w[a * r_prev + ap];
                        }
                        tmp[b * r_prev + ap] = s;
                    }
                }
                // w_new[b, b'] += sum_{a'} tmp[b, a'] * G[a', i, b']
                for b in 0..r_k {
                    for bp in 0..r_k {
                        let mut s = 0.0f32;
                        for ap in 0..r_prev {
                            s += tmp[b * r_prev + ap] * g[[ap, i, bp]];
                        }
                        w_new[b * r_k + bp] += s;
                    }
                }
            }
            w = w_new;
            w_dim = r_k;
        }
        w[0]
    }

    /// Re-round to (max_rank, tol). v0.1 path: reconstruct → re-decompose.
    /// v0.2 will replace this with the standard right-to-left QR sweep
    /// followed by left-to-right truncated SVD sweep — O(d · n · r³) instead
    /// of O(n^d).
    pub fn round(&mut self, max_rank: usize, tol: f32) -> Result<(), TtError> {
        let dense = self.to_dense();
        *self = Self::from_dense(dense.view(), max_rank, tol)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, IxDyn};

    fn approx(a: f32, b: f32, eps: f32) -> bool { (a - b).abs() < eps }

    #[test]
    fn roundtrip_random_tensor() {
        // Dense → TT (full rank) → dense should be bit-close.
        let shape = [3, 4, 5];
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32).sin() * 0.5).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), data.clone()).unwrap();
        let tt = TtTensor::from_dense(arr.view(), 64, 1e-12).unwrap();
        let back = tt.to_dense();
        for (a, b) in data.iter().zip(back.iter()) {
            assert!(approx(*a, *b, 1e-5), "{} vs {}", a, b);
        }
    }

    #[test]
    fn rank1_tensor_compresses() {
        // Outer product of three vectors is exact rank-1 in TT.
        let a = [1.0f32, 2.0, 3.0];
        let b = [0.5f32, -0.25];
        let c = [2.0f32, 1.0, -1.0, 0.5];
        let mut data = Vec::with_capacity(3 * 2 * 4);
        for &ai in &a { for &bi in &b { for &ci in &c { data.push(ai * bi * ci); } } }
        let arr = Array::from_shape_vec(IxDyn(&[3, 2, 4]), data.clone()).unwrap();
        let tt = TtTensor::from_dense(arr.view(), 16, 1e-10).unwrap();
        let ranks = tt.ranks();
        assert_eq!(ranks, vec![1, 1, 1, 1], "expected all TT-ranks = 1, got {:?}", ranks);
        let back = tt.to_dense();
        for (x, y) in data.iter().zip(back.iter()) {
            assert!(approx(*x, *y, 1e-5));
        }
    }

    #[test]
    fn eval_matches_dense() {
        let shape = [4, 3, 5];
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i * 7 + 1) as f32).ln()).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), data).unwrap();
        let tt = TtTensor::from_dense(arr.view(), 32, 1e-12).unwrap();
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let from_tt = tt.eval(&[i, j, k]).unwrap();
                    let from_dense = arr[[i, j, k]];
                    assert!(approx(from_tt, from_dense, 1e-4),
                            "[{i},{j},{k}]: {from_tt} vs {from_dense}");
                }
            }
        }
    }

    #[test]
    fn norm_sq_matches_dense() {
        let shape = [3, 4, 3];
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.13).cos()).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), data.clone()).unwrap();
        let dense_norm_sq: f32 = data.iter().map(|v| v * v).sum();
        let tt = TtTensor::from_dense(arr.view(), 32, 1e-12).unwrap();
        let tt_norm_sq = tt.norm_sq();
        assert!(approx(tt_norm_sq, dense_norm_sq, 1e-3),
                "norm² mismatch: tt={tt_norm_sq}, dense={dense_norm_sq}");
    }

    #[test]
    fn truncation_reduces_rank() {
        let shape = [6, 6, 6];
        let n: usize = shape.iter().product();
        // Build a tensor that is exactly rank-2 in TT form.
        // T = u1⊗v1⊗w1 + u2⊗v2⊗w2
        let u1 = [1.0f32, 2.0, 1.5, -1.0, 0.5, 0.25];
        let v1 = [0.5f32, -0.5, 1.0, 0.2, -0.1, 0.3];
        let w1 = [1.0f32, 0.0, -1.0, 0.5, -0.5, 0.25];
        let u2 = [0.1f32, -0.2, 0.3, 0.7, -0.4, 0.1];
        let v2 = [1.0f32, 0.0, 0.5, -0.5, 0.25, -0.25];
        let w2 = [-1.0f32, 0.5, 0.0, 0.5, -0.5, 1.0];
        let mut data = Vec::with_capacity(n);
        for i in 0..6 { for j in 0..6 { for k in 0..6 {
            data.push(u1[i]*v1[j]*w1[k] + u2[i]*v2[j]*w2[k]);
        }}}
        let arr = Array::from_shape_vec(IxDyn(&shape), data).unwrap();
        // tol = 1e-5: tighter would expose f32 build-noise as spurious rank.
        let tt = TtTensor::from_dense(arr.view(), 8, 1e-5).unwrap();
        let ranks = tt.ranks();
        // Exact algebraic rank is 2 in both inner positions; we allow the
        // numerical rank to slip to 2 cleanly.
        assert!(ranks[1] <= 2, "expected r_1 ≤ 2, got {}", ranks[1]);
        assert!(ranks[2] <= 2, "expected r_2 ≤ 2, got {}", ranks[2]);
        // And the compression ratio should reflect the rank-2 structure.
        // dense = 216 f32, tt at max rank 2 ≤ 1*6*2 + 2*6*2 + 2*6*1 = 48 → ratio ≥ 4×.
        assert!(tt.compression_ratio() >= 4.0,
                "expected ≥4× compression, got {:.2}×", tt.compression_ratio());
    }
}
