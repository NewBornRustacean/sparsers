use ndarray::{Array2, ArrayView2, Axis};
use sparsers::{
    core::{
        common::{Index, Scalar, SparseMatrix},
        csr::CsrContainer,
        kernel::{sddmm, spmm_dense},
    },
    interface::error::{SparsersError, SparsersResult},
};

/// Perform Sparse-Dense Matrix Multiplication (SpMM) where the sparse matrix is in ndarray form.
/// The dense matrix is also in ndarray.
pub fn spmm_ndarray<I: Index, V: Scalar>(
    sparse: &ArrayView2<V>,
    dense: &ArrayView2<V>,
) -> SparsersResult<Array2<V>, SparsersError> {
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
        dense.to_standard_layout()
    };

    let dense_slice = dense_view.as_slice().expect("Dense matrix must be contiguous");
    let csr_matrix: CsrContainer<I, V> = CsrContainer::from_ndarray(sparse);

    let mut output = Array2::<V>::zeros((sparse_row, dense_col));
    let output_slice = output.as_slice_mut().expect("Output matrix must be contiguous");

    csr_matrix
        .par_zip_out(output_slice, PartitionStrategy::Fixed(dense_cols))
        .for_each(|(_row_idx, col_indices, a_values, out_row_slice)| {
            // Inner Kernel: SpMM Vector-Dot Product
            for (&col_idx, &a_val) in col_indices.iter().zip(a_values.iter()) {
                let k = col_idx.to_usize();
                let b_start = k * dense_cols;

                let b_row = &dense_slice[b_start..b_start + dense_cols];

                for (out_val, &b_val) in out_row_slice.iter_mut().zip(b_row.iter()) {
                    *out_val += a_val * b_val;
                }
            }
        });

    Ok(output)
}
