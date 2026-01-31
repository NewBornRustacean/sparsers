use rayon::prelude::*;

use crate::{
    core::common::{Index, MutableSparseMatrix, Scalar, SparseMatrix},
    csr::{CsrView, CsrViewMut},
};

/// Sparse Matrix-Matrix Multiplication (SpMM) where the second matrix is dense.
///
/// # Arguments
/// - `a`: Sparse matrix in CSR format.
/// - `b`: Dense matrix stored in row-major order.
/// - `c`: Output buffer for the result matrix, also in row-major order.
/// - `b_cols`: Number of columns in the dense matrix `b`.
///
/// # Panics
/// Panics if the dimensions of the matrices do not align for multiplication or if the output buffer
/// size does not match the expected size.
pub fn spmm_dense<M, I, V>(a: &M, b: &[V], c: &mut [V], b_cols: usize)
where
    M: SparseMatrix<I, V> + Sync,
    I: Index,
    V: Scalar,
{
    let (a_rows, a_cols) = a.shape();

    assert_eq!(c.len(), a_rows * b_cols, "Output buffer size mismatch");
    assert!(b.len() >= a_cols * b_cols, "Dense matrix B is too small");

    c.fill(V::zero());

    c.par_chunks_mut(b_cols).enumerate().for_each(|(i, c_row)| {
        if let Some((a_col_indices, a_values)) = a.row(i) {
            for (&col_index, &val) in a_col_indices.iter().zip(a_values.iter()) {
                let b_row_start = col_index.to_usize() * b_cols;
                let b_row_end = b_row_start + b_cols;
                let b_row = &b[b_row_start..b_row_end];

                // Inner Loop: Dense-Vector Accumulation
                for (c_out, &b_val) in c_row.iter_mut().zip(b_row.iter()) {
                    *c_out += val * b_val;
                }
            }
        }
    });
}

/// Sampled Dense-Dense Matrix Multiplication (SDDMM).
///
/// Computes: S_ij = S_ij * (D1_i * D2_j^T)
/// Where (i, j) are the indices of non-zero elements in the sparse matrix S.
///
/// # Arguments
/// * `s` - Mutable view of the sparse matrix in CSR format.
/// * `d1` - First dense matrix (M x K), stored in row-major order.
/// * `d2` - Second dense matrix (N x K), where each row j represents the vector to dot with D1_i.
///          Note: D2 is effectively pre-transposed for optimal cache locality.
/// * `k` - The inner dimension (latent factor size).
///
/// # Panics
/// Panics if the dimensions of `d1` or `d2` do not match the shape of `s` and `k`.
pub fn sddmm<'a, M, I, V>(s: M, d1: &[V], d2: &[V], k: usize)
where
    M: MutableSparseMatrix<'a, I, V> + Sync,
    I: Index,
    V: Scalar,
{
    let rows = s.split_into_rows_mut();

    rows.into_par_iter().for_each(|(i, col_indices, row_values)| {
        let d1_row = &d1[i * k..(i + 1) * k];

        for (idx, &col_idx_raw) in col_indices.iter().enumerate() {
            let j = col_idx_raw.to_usize();
            let d2_row = &d2[j * k..(j + 1) * k];

            let mut dot = V::zero();
            for p in 0..k {
                dot += d1_row[p] * d2_row[p];
            }

            row_values[idx] *= dot;
        }
    });
}

#[cfg(test)]
mod kernel_tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::csr::{CsrContainer, CsrView};

    ////////////////////////////////
    /// Test cases for spmm_dense///
    ////////////////////////////////
    #[test]
    fn test_spmm_dense_happy_path() {
        // A (2x3): [[1, 0, 2], [0, 0, 3]]
        // B (3x2): [[1, 2], [3, 4], [5, 6]]
        // C = A * B = [[11, 14], [15, 18]]

        let row_ptr = vec![0u32, 2, 3];
        let col_idx = vec![0u32, 2, 2];
        let vals = vec![1.0f32, 2.0, 3.0];
        let a = CsrView::new((2, 3), &row_ptr, &col_idx, &vals);

        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = vec![0.0f32; 2 * 2];

        spmm_dense(&a, &b, &mut c, 2);

        assert_eq!(c, vec![11.0, 14.0, 15.0, 18.0]);
    }

    #[test]
    fn test_spmm_dense_empty_row() {
        // A (3x2): [[1, 2], [0, 0], [3, 4]] -> Middle row is empty
        // B (2x1): [[10], [20]]
        // Expected C: [[50], [0], [110]]

        let row_ptr = vec![0u32, 2, 2, 4]; // row 1: 2 to 2 (empty)
        let col_idx = vec![0u32, 1, 0, 1];
        let vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let a = CsrView::new((3, 2), &row_ptr, &col_idx, &vals);

        let b = vec![10.0f32, 20.0];
        let mut c = vec![0.0f32; 3 * 1];

        spmm_dense(&a, &b, &mut c, 1);

        assert_eq!(c, vec![50.0, 0.0, 110.0]);
    }

    #[test]
    fn test_spmm_dense_padding_safety() {
        // verify that spmm_dense only reads up to b_cols in B rows
        let row_ptr = vec![0u32, 1];
        let col_idx = vec![0u32];
        let vals = vec![2.0f32];
        let a = CsrView::new((1, 1), &row_ptr, &col_idx, &vals);

        let b = vec![10.0f32, 99.0, 99.0]; // 99.0 are noise values
        let mut c = vec![0.0f32; 1];

        // confirm that only the 1st element of each row in B is read
        spmm_dense(&a, &b, &mut c, 1);

        assert_eq!(c[0], 20.0);
    }

    ////////////////////////////
    /// Test cases for sddmm ///
    ////////////////////////////
    #[test]
    fn test_sddmm_basic_correctness() {
        // S = [1.0, 0.0]  (2x2 matrix)
        //     [0.0, 2.0]
        let row_ptrs = vec![0u32, 1, 2];
        let col_indices = vec![0u32, 1];
        let values = vec![1.0f32, 1.0];
        let mut s = CsrContainer {
            row_ptrs,
            col_indices,
            values: values.clone(),
            shape: (2, 2),
        };

        // D1 = [1.0, 2.0] (2x2, K=2)
        //      [3.0, 4.0]
        let d1 = vec![1.0f32, 2.0, 3.0, 4.0];

        // D2 = [1.0, 1.0] (2x2, K=2) -> D2_j means row j
        //      [0.0, 1.0]
        let d2 = vec![1.0f32, 1.0, 0.0, 1.0];
        let k = 2;

        let view_mut = CsrViewMut::new(s.shape, &s.row_ptrs, &s.col_indices, &mut s.values);

        sddmm(view_mut, &d1, &d2, k);

        // S[0,0] = S[0,0] * (D1[0,:] ⋅ D2[0,:]) = 1.0 * (1*1 + 2*1) = 3.0
        // S[1,1] = S[1,1] * (D1[1,:] ⋅ D2[1,:]) = 1.0 * (3*0 + 4*1) = 4.0
        assert_relative_eq!(s.values[0], 3.0);
        assert_relative_eq!(s.values[1], 4.0);
    }

    #[test]
    fn test_sddmm_with_empty_row() {
        // S = [0, 0] (1st row is empty)
        //     [1, 0] (2nd row has one element at col 0)
        let row_ptrs = vec![0u32, 0, 1];
        let col_indices = vec![0u32];
        let values = vec![5.0f32];
        let mut s = CsrContainer {
            row_ptrs,
            col_indices,
            values,
            shape: (2, 2),
        };

        let d1 = vec![1.0, 1.0, 2.0, 2.0];
        let d2 = vec![3.0, 3.0, 4.0, 4.0];
        let k = 2;

        let view_mut = CsrViewMut::new(s.shape, &s.row_ptrs, &s.col_indices, &mut s.values);

        // should pass even if there's a empty row
        sddmm(view_mut, &d1, &d2, k);

        // S[1,0] = 5.0 * (D1[1,:] ⋅ D2[0,:]) = 5.0 * (2*3 + 2*3) = 5.0 * 12.0 = 60.0
        assert_relative_eq!(s.values[0], 60.0);
    }

    #[test]
    fn test_sddmm_u64_index() {
        let row_ptrs = vec![0u64, 1];
        let col_indices = vec![0u64];
        let values = vec![2.0f32];
        let mut s = CsrContainer {
            row_ptrs,
            col_indices,
            values,
            shape: (1, 1),
        };

        let d1 = vec![1.5f32];
        let d2 = vec![2.0f32];
        let k = 1;

        let view_mut = CsrViewMut::new(s.shape, &s.row_ptrs, &s.col_indices, &mut s.values);

        sddmm(view_mut, &d1, &d2, k);

        // 2.0 * (1.5 * 2.0) = 6.0
        assert_relative_eq!(s.values[0], 6.0);
    }
}
