use ndarray::{Array2, ArrayView2, Axis};

use crate::{
    core::{
        common::{Index, PartitionStrategy, Scalar, SparseMatrix},
        csr::CsrContainer,
        kernel::{sddmm, spmm_dense},
    },
    interface::error::{SparsersError, SparsersResult},
};

/// Interface function to perform Sparse-Dense Matrix Multiplication (SpMM)
/// Perform Sparse-Dense Matrix Multiplication (SpMM) where the sparse matrix is in ndarray form.
pub fn spmm_ndarray<I: Index, V: Scalar>(
    sparse: ArrayView2<V>,
    dense: ArrayView2<V>,
) -> SparsersResult<Array2<V>> {
    let (sparse_row, sparse_col) = sparse.dim();
    let (dense_row, dense_col) = dense.dim();

    if sparse_col != dense_row {
        return Err(SparsersError::DimensionMismatch {
            sparse_cols: sparse_col,
            dense_rows: dense_row,
        });
    }

    let dense_view: ArrayView2<V> = if dense.is_standard_layout() {
        dense.view()
    } else {
        return Err(SparsersError::LayoutError(
            "Dense matrix must be in row-major (C-contiguous) layout".to_string(),
        ));
    };

    let dense_slice = dense_view.as_slice().expect("Dense matrix must be contiguous");
    let csr_matrix: CsrContainer<I, V> = CsrContainer::from_ndarray(sparse);

    let mut output = Array2::<V>::zeros((sparse_row, dense_col));
    let output_slice = output.as_slice_mut().expect("Output matrix must be contiguous");
    spmm_dense(&csr_matrix, dense_slice, output_slice, dense_col);

    Ok(output)
}

/// Interface function to perform Sampled Dense-Dense Matrix Multiplication (SDDMM)
/// Perform Sampled Dense-Dense Matrix Multiplication (SDDMM) where the sparse matrix
pub fn sddmm_ndarray<I: Index, V: Scalar>(
    sparse: ArrayView2<V>,
    dense1: ArrayView2<V>,
    dense2: ArrayView2<V>,
) -> SparsersResult<Array2<V>> {
    let (sparse_row, sparse_col) = sparse.dim();
    let (dense1_row, dense1_col) = dense1.dim();
    let (dense2_row, dense2_col) = dense2.dim();

    if sparse_row != dense1_row || sparse_col != dense2_row {
        return Err(SparsersError::DimensionMismatch {
            sparse_cols: sparse_col,
            dense_rows: dense2_row,
        });
    }

    if dense1_col != dense2_col {
        return Err(SparsersError::DimensionMismatch {
            sparse_cols: dense1_col,
            dense_rows: dense2_col,
        });
    }

    let dense1_view: ArrayView2<V> = if dense1.is_standard_layout() {
        dense1.view()
    } else {
        return Err(SparsersError::LayoutError(
            "Dense matrix 1 must be in row-major (C-contiguous) layout".to_string(),
        ));
    };

    let dense2_view: ArrayView2<V> = if dense2.is_standard_layout() {
        dense2.view()
    } else {
        return Err(SparsersError::LayoutError(
            "Dense matrix 2 must be in row-major (C-contiguous) layout".to_string(),
        ));
    };

    let dense1_slice = dense1_view.as_slice().expect("Dense matrix 1 must be contiguous");
    let dense2_slice = dense2_view.as_slice().expect("Dense matrix 2 must be contiguous");

    let csr_matrix: CsrContainer<I, V> = CsrContainer::from_ndarray(sparse);

    let mut output = Array2::<V>::zeros((sparse_row, sparse_col));
    let output_slice = output.as_slice_mut().expect("Output matrix must be contiguous");

    sddmm(
        &csr_matrix,
        dense1_slice,
        dense2_slice,
        output_slice,
        dense1_col,
    );

    Ok(output)
}
