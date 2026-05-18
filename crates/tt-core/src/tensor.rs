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

    /// Re-round to (max_rank, tol). O(d · n_max · r³).
    ///
    /// Two sweeps over the cores:
    ///   1. Right-to-left QR pass orthogonalises each core's rows so that
    ///      the right unfoldings are isometries. This pushes all the
    ///      "weight" of the tensor into the left side.
    ///   2. Left-to-right truncated SVD pass: each core's left unfolding
    ///      is SVD-decomposed; singular values below `tol * σ_max` or
    ///      beyond `max_rank` are dropped, and the truncation factor is
    ///      pushed into the next core to the right.
    ///
    /// Result is equivalent to a fresh `from_dense` on the same tensor at
    /// the same `(max_rank, tol)`, but never materialises the dense form —
    /// crucial for d > 4 where dense storage would explode.
    pub fn round(&mut self, max_rank: usize, tol: f32) -> Result<(), TtError> {
        if max_rank == 0 { return Err(TtError::InvalidRank { got: 0 }); }
        let d = self.ndim();
        if d <= 1 { return Ok(()); }

        // ----- Pass 1: right-to-left QR orthogonalisation -----
        for k in (1..d).rev() {
            let g = &self.cores[k];
            let (r_pre, n_k, r_post) = (g.shape()[0], g.shape()[1], g.shape()[2]);

            // Unfold G into (r_pre, n_k * r_post) row-major (as f64).
            // Then transpose to (n_k * r_post, r_pre) for the QR.
            let n_cols = n_k * r_post;
            let mt: nalgebra::DMatrix<f64> = nalgebra::DMatrix::from_fn(n_cols, r_pre, |col, row| {
                let ni = col / r_post;
                let rp = col % r_post;
                g[[row, ni, rp]] as f64
            });
            let qr = mt.qr();
            let q = qr.q();          // shape (n_cols, p), p = min(n_cols, r_pre)
            let r = qr.r();          // shape (p, r_pre) upper triangular
            let p = q.ncols();

            // New core[k] = Q^T reshaped to (p, n_k, r_post).
            let mut new_core_k = Array3::<f32>::zeros((p, n_k, r_post));
            for i in 0..p {
                for col in 0..n_cols {
                    let ni = col / r_post;
                    let rp = col % r_post;
                    new_core_k[[i, ni, rp]] = q[(col, i)] as f32;
                }
            }
            self.cores[k] = new_core_k;

            // Absorb R^T (shape r_pre x p) into core[k-1] along its r_post axis.
            // core[k-1] reshape (r_pre^{(-1)} * n_{k-1}, r_pre) becomes
            // (r_pre^{(-1)} * n_{k-1}, p) via right-multiply.
            let prev = &self.cores[k - 1];
            let (rA, n_prev, rB) = (prev.shape()[0], prev.shape()[1], prev.shape()[2]);
            debug_assert_eq!(rB, r_pre);
            let mut new_prev = Array3::<f32>::zeros((rA, n_prev, p));
            for a in 0..rA {
                for np in 0..n_prev {
                    for j in 0..p {
                        let mut s = 0.0f64;
                        for rr in 0..r_pre {
                            s += (prev[[a, np, rr]] as f64) * r[(j, rr)];
                        }
                        new_prev[[a, np, j]] = s as f32;
                    }
                }
            }
            self.cores[k - 1] = new_prev;
        }

        // ----- Pass 2: left-to-right SVD truncation -----
        for k in 0..(d - 1) {
            let g = &self.cores[k];
            let (r_pre, n_k, r_post) = (g.shape()[0], g.shape()[1], g.shape()[2]);

            // Unfold G into (r_pre * n_k, r_post) f64.
            let n_rows = r_pre * n_k;
            let m: nalgebra::DMatrix<f64> = nalgebra::DMatrix::from_fn(n_rows, r_post, |row, col| {
                let a = row / n_k;
                let ni = row % n_k;
                g[[a, ni, col]] as f64
            });
            let svd = m.svd(true, true);
            let u = svd.u.ok_or(TtError::SvdFailed)?;
            let s = svd.singular_values;
            let vt = svd.v_t.ok_or(TtError::SvdFailed)?;

            let sigma_max = s.iter().fold(0.0f64, |a, &b| a.max(b));
            let cutoff = (tol as f64) * sigma_max;
            let mut keep = s.iter().filter(|&&v| v > cutoff).count();
            keep = keep.min(max_rank).min(s.len()).max(1);

            // New core[k] = U[:, :keep] reshaped (r_pre, n_k, keep).
            let mut new_g = Array3::<f32>::zeros((r_pre, n_k, keep));
            for a in 0..r_pre {
                for ni in 0..n_k {
                    let row = a * n_k + ni;
                    for j in 0..keep {
                        new_g[[a, ni, j]] = u[(row, j)] as f32;
                    }
                }
            }
            self.cores[k] = new_g;

            // Absorb diag(s[..keep]) @ V^T[..keep, :] into core[k+1] along r_pre.
            // factor shape: (keep, r_post)
            let next = &self.cores[k + 1];
            let (rA, n_next, rB) = (next.shape()[0], next.shape()[1], next.shape()[2]);
            debug_assert_eq!(rA, r_post);
            let mut new_next = Array3::<f32>::zeros((keep, n_next, rB));
            for i in 0..keep {
                let sigma = s[i];
                for nn in 0..n_next {
                    for b in 0..rB {
                        let mut acc = 0.0f64;
                        for r_in in 0..r_post {
                            acc += sigma * vt[(i, r_in)] * (next[[r_in, nn, b]] as f64);
                        }
                        new_next[[i, nn, b]] = acc as f32;
                    }
                }
            }
            self.cores[k + 1] = new_next;
        }

        Ok(())
    }

    /// Element-wise sum T = A + B. Both operands must share `shape`.
    ///
    /// Construction: interior cores are formed by block-diagonal stacking,
    /// so the resulting TT-rank at each interior edge is the sum of the
    /// two operands' ranks. The first core concatenates along `r_post`
    /// (the only available rank); the last core concatenates along
    /// `r_pre`. The result is mathematically exact; call `round()` after
    /// to truncate the (now larger) ranks back to something useful.
    pub fn add(&self, other: &Self) -> Result<Self, TtError> {
        if self.shape != other.shape {
            return Err(TtError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        let d = self.ndim();
        let mut new_cores = Vec::with_capacity(d);

        for k in 0..d {
            let a = &self.cores[k];
            let b = &other.cores[k];
            let (ra_pre, n_k, ra_post) = (a.shape()[0], a.shape()[1], a.shape()[2]);
            let (rb_pre, n_k_b, rb_post) = (b.shape()[0], b.shape()[1], b.shape()[2]);
            debug_assert_eq!(n_k, n_k_b, "physical mode mismatch at core {k}");

            if d == 1 {
                // Single-mode tensor: both cores are (1, n, 1). Plain add.
                let mut c = Array3::<f32>::zeros((1, n_k, 1));
                for i in 0..n_k { c[[0, i, 0]] = a[[0, i, 0]] + b[[0, i, 0]]; }
                new_cores.push(c);
            } else if k == 0 {
                // (1, n_k, ra_post + rb_post): concat along r_post.
                let r_post = ra_post + rb_post;
                let mut c = Array3::<f32>::zeros((1, n_k, r_post));
                for i in 0..n_k {
                    for j in 0..ra_post { c[[0, i, j]]            = a[[0, i, j]]; }
                    for j in 0..rb_post { c[[0, i, ra_post + j]]  = b[[0, i, j]]; }
                }
                new_cores.push(c);
            } else if k == d - 1 {
                // (ra_pre + rb_pre, n_k, 1): concat along r_pre.
                let r_pre = ra_pre + rb_pre;
                let mut c = Array3::<f32>::zeros((r_pre, n_k, 1));
                for i in 0..n_k {
                    for j in 0..ra_pre { c[[j, i, 0]]              = a[[j, i, 0]]; }
                    for j in 0..rb_pre { c[[ra_pre + j, i, 0]]     = b[[j, i, 0]]; }
                }
                new_cores.push(c);
            } else {
                // Interior core: block-diagonal across both rank axes.
                let r_pre  = ra_pre  + rb_pre;
                let r_post = ra_post + rb_post;
                let mut c = Array3::<f32>::zeros((r_pre, n_k, r_post));
                for ip in 0..ra_pre {
                    for i in 0..n_k {
                        for jp in 0..ra_post {
                            c[[ip, i, jp]] = a[[ip, i, jp]];
                        }
                    }
                }
                for ip in 0..rb_pre {
                    for i in 0..n_k {
                        for jp in 0..rb_post {
                            c[[ra_pre + ip, i, ra_post + jp]] = b[[ip, i, jp]];
                        }
                    }
                }
                new_cores.push(c);
            }
        }

        Ok(Self { cores: new_cores, shape: self.shape.clone() })
    }

    /// In-place scalar multiplication T ← s · T. Cost is O(first core size);
    /// only one core needs scaling because the chain is multilinear.
    pub fn scale(&mut self, s: f32) {
        if let Some(c) = self.cores.first_mut() {
            for v in c.iter_mut() { *v *= s; }
        }
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
    fn add_matches_dense() {
        let shape = [4, 5, 6];
        let n: usize = shape.iter().product();
        let a_v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.13).sin()).collect();
        let b_v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.31).cos() * 0.5).collect();
        let sum_v: Vec<f32> = a_v.iter().zip(b_v.iter()).map(|(a, b)| a + b).collect();

        let a_arr   = Array::from_shape_vec(IxDyn(&shape), a_v.clone()).unwrap();
        let b_arr   = Array::from_shape_vec(IxDyn(&shape), b_v.clone()).unwrap();
        let sum_arr = Array::from_shape_vec(IxDyn(&shape), sum_v.clone()).unwrap();

        let tt_a = TtTensor::from_dense(a_arr.view(), 32, 1e-12).unwrap();
        let tt_b = TtTensor::from_dense(b_arr.view(), 32, 1e-12).unwrap();
        let tt_sum = tt_a.add(&tt_b).unwrap();

        // Ranks should sum at the interior edge (or be ≤ the algebraic limit).
        let ra = tt_a.ranks();
        let rb = tt_b.ranks();
        let rs = tt_sum.ranks();
        for k in 1..shape.len() {
            assert_eq!(rs[k], ra[k] + rb[k], "rank at edge {} should be sum", k);
        }

        // Reconstruction matches dense element-wise.
        let back = tt_sum.to_dense();
        for (a, b) in sum_arr.iter().zip(back.iter()) {
            assert!(approx(*a, *b, 1e-4), "sum mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn round_recovers_rank_after_add_self() {
        // T + T should equal 2T. After round, ranks should match T's, not 2× T's.
        let shape = [4, 5, 6];
        let n: usize = shape.iter().product();
        let a_v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.17).sin() + 0.3).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), a_v).unwrap();
        let tt = TtTensor::from_dense(arr.view(), 32, 1e-12).unwrap();
        let orig_ranks = tt.ranks();

        let mut summed = tt.add(&tt).unwrap();
        let pre_round = summed.ranks();
        // ranks should have doubled at interior edges
        for k in 1..shape.len() {
            assert_eq!(pre_round[k], orig_ranks[k] * 2);
        }
        summed.round(32, 1e-8).unwrap();
        let post_round = summed.ranks();
        // After round, ranks should be back at or below the originals.
        for k in 1..shape.len() {
            assert!(post_round[k] <= orig_ranks[k] + 1,
                "edge {}: orig={}, after round={}", k, orig_ranks[k], post_round[k]);
        }
        // And the values should be 2× the original.
        let dense = summed.to_dense();
        let orig_dense = tt.to_dense();
        for (a, b) in orig_dense.iter().zip(dense.iter()) {
            assert!(approx(*b, 2.0 * *a, 1e-3),
                "expected 2x, got {} vs {}", b, 2.0 * a);
        }
    }

    #[test]
    fn add_then_round_recovers_compact_form() {
        // (a + (-a)) should round to (almost) zero.
        let shape = [4, 5, 6];
        let n: usize = shape.iter().product();
        let a_v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.13).sin()).collect();
        let a_arr = Array::from_shape_vec(IxDyn(&shape), a_v).unwrap();
        let tt_a = TtTensor::from_dense(a_arr.view(), 32, 1e-12).unwrap();
        let mut tt_neg = tt_a.clone();
        tt_neg.scale(-1.0);
        let mut sum = tt_a.add(&tt_neg).unwrap();
        sum.round(8, 1e-6).unwrap();
        let back = sum.to_dense();
        for v in back.iter() {
            assert!(v.abs() < 1e-4, "expected near-zero, got {v}");
        }
    }

    #[test]
    fn scale_is_multiplicative() {
        let shape = [3, 4, 5];
        let n: usize = shape.iter().product();
        let a_v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), a_v.clone()).unwrap();
        let mut tt = TtTensor::from_dense(arr.view(), 32, 1e-12).unwrap();
        tt.scale(3.0);
        let back = tt.to_dense();
        for (orig, scaled) in a_v.iter().zip(back.iter()) {
            assert!(approx(scaled / orig, 3.0, 1e-3),
                "expected 3× scale, got {} / {} = {}", scaled, orig, scaled / orig);
        }
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
