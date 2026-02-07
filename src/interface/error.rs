use thiserror::Error;

#[derive(Error, Debug)]
pub enum SparsersError {
    #[error("Dimension mismatch: sparse.cols ({sparse_cols}) != dense.rows ({dense_rows})")]
    DimensionMismatch {
        sparse_cols: usize,
        dense_rows: usize,
    },

    #[error("Memory layout error: {0}")]
    LayoutError(String),

    #[error("Index out of bounds: row {row}, col {col}")]
    IndexOutOfBounds { row: usize, col: usize },

    #[error("Internal engine error: {0}")]
    Internal(String),
}

pub type SparsersResult<T> = Result<T, SparsersError>;
