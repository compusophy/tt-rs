use thiserror::Error;

#[derive(Error, Debug)]
pub enum TtError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("dimension mismatch: tensor has {ndim} dims, got {got} indices")]
    IndexDimMismatch { ndim: usize, got: usize },

    #[error("index out of bounds: dim {dim} has size {size}, got {got}")]
    IndexOutOfBounds { dim: usize, size: usize, got: usize },

    #[error("SVD did not converge")]
    SvdFailed,

    #[error("rank must be at least 1, got {got}")]
    InvalidRank { got: usize },

    #[error("at least 1 mode required, got 0")]
    EmptyTensor,
}
