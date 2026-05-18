//! TT-matrix — a matrix M of shape (∏m_k, ∏n_k) stored as a TT-tensor over
//! the interleaved index layout (m_1, n_1, m_2, n_2, …, m_d, n_d).
//!
//! The headline operation is `matvec`: y = M x. The TT-aware implementation
//! never materializes the dense matrix — it walks the cores left-to-right,
//! contracting against `x`'s reshaped form one mode at a time. Memory and
//! compute are both polynomial in the TT-rank rather than exponential in d.
//!
//! Convention: 2d cores total, alternating row-core (over m_k) and col-core
//! (over n_k). cores[2k]   shape = (r_{2k},     m_k, r_{2k+1})
//!             cores[2k+1] shape = (r_{2k+1},   n_k, r_{2k+2})

use ndarray::{ArrayD, ArrayView2, ArrayViewD, IxDyn};

use crate::error::TtError;
use crate::tensor::TtTensor;

#[derive(Debug, Clone)]
pub struct TtMatrix {
    /// TT-tensor over the interleaved shape (m_1, n_1, m_2, n_2, …, m_d, n_d).
    pub tt: TtTensor,
    pub m_shape: Vec<usize>,
    pub n_shape: Vec<usize>,
}

impl TtMatrix {
    pub fn d(&self) -> usize { self.m_shape.len() }
    pub fn rows(&self) -> usize { self.m_shape.iter().product() }
    pub fn cols(&self) -> usize { self.n_shape.iter().product() }

    /// Build a TT-matrix from pre-constructed cores. Lets you create matrices
    /// whose dense form would not fit in RAM. The 2d cores must alternate
    /// row-mode / col-mode and chain correctly: cores[k].shape() = (r_{k-1},
    /// mode_k, r_k) with r_0 = r_{2d} = 1, mode_2k = m_k, mode_{2k+1} = n_k.
    pub fn from_cores(
        m_shape: &[usize],
        n_shape: &[usize],
        cores: Vec<ndarray::Array3<f32>>,
    ) -> Result<Self, TtError> {
        if m_shape.len() != n_shape.len() {
            return Err(TtError::ShapeMismatch {
                expected: m_shape.to_vec(),
                got: n_shape.to_vec(),
            });
        }
        let d = m_shape.len();
        if cores.len() != 2 * d {
            return Err(TtError::ShapeMismatch {
                expected: vec![2 * d],
                got: vec![cores.len()],
            });
        }
        // Validate chain + interleaving.
        let mut r_prev = 1usize;
        for (k, core) in cores.iter().enumerate() {
            let (r_pre, mode, r_post) = (core.shape()[0], core.shape()[1], core.shape()[2]);
            if r_pre != r_prev {
                return Err(TtError::ShapeMismatch {
                    expected: vec![r_prev, mode, r_post],
                    got: vec![r_pre, mode, r_post],
                });
            }
            let expected_mode = if k % 2 == 0 { m_shape[k / 2] } else { n_shape[k / 2] };
            if mode != expected_mode {
                return Err(TtError::ShapeMismatch {
                    expected: vec![r_pre, expected_mode, r_post],
                    got: vec![r_pre, mode, r_post],
                });
            }
            r_prev = r_post;
        }
        if r_prev != 1 {
            return Err(TtError::InvalidRank { got: r_prev });
        }

        // Build the interleaved TT-tensor shape.
        let interleaved_shape: Vec<usize> = m_shape
            .iter()
            .zip(n_shape.iter())
            .flat_map(|(&m, &n)| [m, n])
            .collect();
        let tt = TtTensor { cores, shape: interleaved_shape };
        Ok(Self { tt, m_shape: m_shape.to_vec(), n_shape: n_shape.to_vec() })
    }

    /// Construct a TT-matrix from a dense (rows, cols) matrix by interleaving
    /// the row and column multi-indices, then running TT-SVD.
    pub fn from_dense(
        m_shape: &[usize],
        n_shape: &[usize],
        data: ArrayView2<f32>,
        max_rank: usize,
        tol: f32,
    ) -> Result<Self, TtError> {
        if m_shape.len() != n_shape.len() {
            return Err(TtError::ShapeMismatch {
                expected: m_shape.to_vec(),
                got:      n_shape.to_vec(),
            });
        }
        let d = m_shape.len();
        let m_rows: usize = m_shape.iter().product();
        let m_cols: usize = n_shape.iter().product();
        if data.shape() != [m_rows, m_cols] {
            return Err(TtError::ShapeMismatch {
                expected: vec![m_rows, m_cols],
                got:      data.shape().to_vec(),
            });
        }

        // Interleaved shape: (m_1, n_1, m_2, n_2, …, m_d, n_d).
        let interleaved_shape: Vec<usize> = m_shape
            .iter()
            .zip(n_shape.iter())
            .flat_map(|(&m, &n)| [m, n])
            .collect();

        let total: usize = interleaved_shape.iter().product();
        debug_assert_eq!(total, m_rows * m_cols);

        let mut buf = vec![0.0f32; total];
        for r in 0..m_rows {
            let i_idx = decompose(r, m_shape);
            for c in 0..m_cols {
                let j_idx = decompose(c, n_shape);
                // Linear index in the interleaved layout (row-major over
                // m_1, n_1, m_2, n_2, …).
                let mut linear = 0usize;
                for k in 0..d {
                    linear = linear * m_shape[k] + i_idx[k];
                    linear = linear * n_shape[k] + j_idx[k];
                }
                buf[linear] = data[[r, c]];
            }
        }

        let arr = ArrayD::from_shape_vec(IxDyn(&interleaved_shape), buf)
            .map_err(|_| TtError::SvdFailed)?;
        let tt = TtTensor::from_dense(arr.view(), max_rank, tol)?;
        debug_assert_eq!(tt.ndim(), 2 * d);

        Ok(Self { tt, m_shape: m_shape.to_vec(), n_shape: n_shape.to_vec() })
    }

    /// y = M x, computed by walking the TT cores left-to-right and contracting
    /// against the reshaped `x` one mode at a time. The intermediate tensor
    /// grows on the m-side and shrinks on the n-side as it goes.
    ///
    /// Complexity: O(d · r² · max(m, n) · ∏_other) — polynomial in TT-rank.
    /// The dense reconstruction is never built.
    pub fn matvec(&self, x: ArrayViewD<f32>) -> Result<ArrayD<f32>, TtError> {
        if x.shape() != self.n_shape.as_slice() {
            return Err(TtError::ShapeMismatch {
                expected: self.n_shape.clone(),
                got:      x.shape().to_vec(),
            });
        }
        let d = self.d();

        // Working buffer + shape. Initial: leading axis of size 1 for the
        // left-dangling rank, then all n-modes.
        let mut w_shape: Vec<usize> =
            std::iter::once(1usize).chain(self.n_shape.iter().copied()).collect();
        let mut w: Vec<f32> = x.iter().copied().collect();

        // Number of m-modes already materialized (live on the *left* of w_shape).
        let mut m_done = 0usize;

        for core_idx in 0..(2 * d) {
            let core = &self.tt.cores[core_idx];
            let r_pre = core.shape()[0];
            let mode  = core.shape()[1];
            let r_post = core.shape()[2];
            let is_row_core = core_idx % 2 == 0;

            if is_row_core {
                // w_shape: (m_1, …, m_{m_done}, r_pre, n_{m_done+1}, …, n_d)
                // Index of r_pre in w_shape: m_done.
                debug_assert_eq!(w_shape[m_done], r_pre,
                    "core {} expects r_pre = {}, w_shape has {}", core_idx, r_pre, w_shape[m_done]);

                let prefix: usize = w_shape[..m_done].iter().product();
                let suffix: usize = w_shape[m_done + 1..].iter().product();
                let m_k = mode;

                let mut new_w = vec![0.0f32; prefix * m_k * r_post * suffix];
                for p in 0..prefix {
                    for m_ix in 0..m_k {
                        for rp in 0..r_post {
                            for s in 0..suffix {
                                let mut acc = 0.0f32;
                                for r in 0..r_pre {
                                    let w_v = w[p * (r_pre * suffix) + r * suffix + s];
                                    let g_v = core[[r, m_ix, rp]];
                                    acc += w_v * g_v;
                                }
                                let idx = p * (m_k * r_post * suffix)
                                        + m_ix * (r_post * suffix)
                                        + rp * suffix
                                        + s;
                                new_w[idx] = acc;
                            }
                        }
                    }
                }

                // w_shape: replace (r_pre) with (m_k, r_post).
                w_shape.remove(m_done);
                w_shape.insert(m_done, m_k);
                w_shape.insert(m_done + 1, r_post);
                w = new_w;
                m_done += 1;
            } else {
                // Col core. w_shape: (m_1, …, m_{m_done}, r_pre, n_next, n_rest…)
                debug_assert_eq!(w_shape[m_done], r_pre);
                debug_assert_eq!(w_shape[m_done + 1], mode);

                let prefix: usize = w_shape[..m_done].iter().product();
                let suffix: usize = w_shape[m_done + 2..].iter().product();
                let n_k = mode;

                let mut new_w = vec![0.0f32; prefix * r_post * suffix];
                for p in 0..prefix {
                    for rp in 0..r_post {
                        for s in 0..suffix {
                            let mut acc = 0.0f32;
                            for r in 0..r_pre {
                                for n_ix in 0..n_k {
                                    let w_v = w[p * (r_pre * n_k * suffix)
                                              + r * (n_k * suffix)
                                              + n_ix * suffix
                                              + s];
                                    let g_v = core[[r, n_ix, rp]];
                                    acc += w_v * g_v;
                                }
                            }
                            new_w[p * (r_post * suffix) + rp * suffix + s] = acc;
                        }
                    }
                }

                // w_shape: replace (r_pre, n_k) with (r_post).
                w_shape.remove(m_done);     // r_pre
                w_shape.remove(m_done);     // n_k (shifted into place)
                w_shape.insert(m_done, r_post);
                w = new_w;
            }
        }

        // Now w_shape is (m_1, …, m_d, 1).
        debug_assert_eq!(w_shape.len(), d + 1);
        debug_assert_eq!(*w_shape.last().unwrap(), 1);
        w_shape.pop();

        ArrayD::from_shape_vec(IxDyn(&w_shape), w).map_err(|_| TtError::SvdFailed)
    }

    /// Total f32 storage of the TT cores.
    pub fn n_params(&self) -> usize {
        self.tt.n_params()
    }

    /// Equivalent dense storage cost.
    pub fn dense_n_params(&self) -> usize {
        self.rows() * self.cols()
    }

    pub fn compression_ratio(&self) -> f32 {
        self.dense_n_params() as f32 / self.n_params().max(1) as f32
    }
}

fn decompose(linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0usize; shape.len()];
    let mut rem = linear;
    for k in (0..shape.len()).rev() {
        idx[k] = rem % shape[k];
        rem /= shape[k];
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, IxDyn};

    fn dense_matvec(m: &Array2<f32>, x: &[f32]) -> Vec<f32> {
        let (rows, cols) = m.dim();
        let mut y = vec![0.0f32; rows];
        for r in 0..rows {
            let mut s = 0.0f32;
            for c in 0..cols {
                s += m[[r, c]] * x[c];
            }
            y[r] = s;
        }
        y
    }

    #[test]
    fn matvec_identity_matches_x() {
        // 8x8 identity, factored as (2,2,2) row modes × (2,2,2) col modes.
        let n = 8;
        let mut data = Array2::<f32>::zeros((n, n));
        for i in 0..n { data[[i, i]] = 1.0; }
        let tm = TtMatrix::from_dense(&[2, 2, 2], &[2, 2, 2], data.view(), 16, 1e-10).unwrap();
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 1.0).collect();
        let x_arr = ndarray::ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), x.clone()).unwrap();
        let y = tm.matvec(x_arr.view()).unwrap();
        // y reshaped flat must equal x.
        let y_flat: Vec<f32> = y.iter().copied().collect();
        for (a, b) in x.iter().zip(y_flat.iter()) {
            assert!((a - b).abs() < 1e-4, "identity matvec mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn matvec_rank1_matches_dense() {
        // Rank-1 matrix: M[r, c] = u[r] * v[c].
        let u = [1.0f32, 2.0, -0.5, 1.5, 0.25, -1.0, 0.7, 0.3];
        let v = [0.4f32, -0.6, 1.1, 0.9, -0.2, 0.5, -0.7, 1.3];
        let mut data = Array2::<f32>::zeros((8, 8));
        for r in 0..8 { for c in 0..8 { data[[r, c]] = u[r] * v[c]; } }
        let tm = TtMatrix::from_dense(&[2, 2, 2], &[2, 2, 2], data.view(), 4, 1e-6).unwrap();
        let x: Vec<f32> = v.iter().map(|x| x + 0.1).collect();
        let x_arr = ndarray::ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), x.clone()).unwrap();
        let y_tt: Vec<f32> = tm.matvec(x_arr.view()).unwrap().iter().copied().collect();
        let y_dense = dense_matvec(&data, &x);
        for (a, b) in y_tt.iter().zip(y_dense.iter()) {
            assert!((a - b).abs() < 1e-3, "rank-1 matvec mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn matvec_random_small_matches_dense() {
        // 4x4 random — small enough that any rank works.
        let data_vec: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.317).sin() * 1.3 + 0.05).collect();
        let data = Array2::from_shape_vec((4, 4), data_vec).unwrap();
        let tm = TtMatrix::from_dense(&[2, 2], &[2, 2], data.view(), 8, 1e-10).unwrap();
        let x_vec: Vec<f32> = vec![0.5, -1.2, 0.7, 0.1];
        let x_arr = ndarray::ArrayD::from_shape_vec(IxDyn(&[2, 2]), x_vec.clone()).unwrap();
        let y_tt: Vec<f32> = tm.matvec(x_arr.view()).unwrap().iter().copied().collect();
        let y_dense = dense_matvec(&data, &x_vec);
        for (a, b) in y_tt.iter().zip(y_dense.iter()) {
            assert!((a - b).abs() < 1e-3, "random matvec mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn from_cores_roundtrips_with_from_dense() {
        // Decompose a small matrix, then rebuild from its own cores → matvec
        // should agree with the original.
        let rows = 16usize;
        let cols = 16usize;
        let v: Vec<f32> = (0..rows*cols).map(|i| ((i as f32) * 0.13).sin()).collect();
        let data = Array2::from_shape_vec((rows, cols), v).unwrap();
        let tm1 = TtMatrix::from_dense(&[4, 4], &[4, 4], data.view(), 8, 1e-8).unwrap();
        let cores_clone = tm1.tt.cores.clone();
        let tm2 = TtMatrix::from_cores(&[4, 4], &[4, 4], cores_clone).unwrap();

        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let x_arr = ndarray::ArrayD::from_shape_vec(IxDyn(&[4, 4]), x).unwrap();
        let y1: Vec<f32> = tm1.matvec(x_arr.view()).unwrap().iter().copied().collect();
        let y2: Vec<f32> = tm2.matvec(x_arr.view()).unwrap().iter().copied().collect();
        for (a, b) in y1.iter().zip(y2.iter()) {
            assert!((a - b).abs() < 1e-6, "from_cores roundtrip mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn compression_ratio_reports_savings() {
        // Build a low-rank 16x16 = 256-element matrix, expect notable savings.
        let u: Vec<f32> = (0..16).map(|i| (i as f32 * 0.1).cos()).collect();
        let v: Vec<f32> = (0..16).map(|j| (j as f32 * 0.07).sin() + 0.3).collect();
        let mut data = Array2::<f32>::zeros((16, 16));
        for r in 0..16 { for c in 0..16 { data[[r, c]] = u[r] * v[c]; } }
        let tm = TtMatrix::from_dense(&[4, 4], &[4, 4], data.view(), 4, 1e-6).unwrap();
        assert!(tm.compression_ratio() >= 2.0,
            "expected ≥2× compression on rank-1 16x16, got {:.2}×",
            tm.compression_ratio());
    }
}
