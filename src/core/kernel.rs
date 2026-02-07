use crate::core::common::{Index, PartitionStrategy, Scalar, SparseMatrix};

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

    a.par_zip_out(c, PartitionStrategy::Fixed(b_cols)).for_each(
        |(_i, a_col_indices, a_values, out_row)| {
            out_row.fill(V::zero());

            for (&col_idx, &a_val) in a_col_indices.iter().zip(a_values.iter()) {
                let b_start = col_idx.to_usize() * b_cols;
                let b_row = &b[b_start..b_start + b_cols];

                for (c_val, &b_val) in out_row.iter_mut().zip(b_row.iter()) {
                    *c_val += a_val * b_val;
                }
            }
        },
    );
}

/// Sampled Dense-Dense Matrix Multiplication (SDDMM).
///
/// Computes: S_ij = S_ij * (D1_i * D2_j^T)
/// Where (i, j) are the indices of non-zero elements in the sparse matrix S.
///
/// # Arguments
/// * `s` - View of the sparse matrix in CSR format.
/// * `d1` - First dense matrix (M x K), stored in row-major order.
/// * `d2` - Second dense matrix (N x K), where each row j represents the vector to dot with D1_i.
///          Note: D2 is effectively pre-transposed for optimal cache locality.
/// * `out` - Output buffer to store the updated non-zero values of the sparse matrix S.
/// * `k` - The inner dimension (latent factor size).
///
/// # Panics
/// Panics if the dimensions of `d1` or `d2` do not match the shape of `s` and `k`.
pub fn sddmm<'a, M, I, V>(s: &M, d1: &[V], d2: &[V], out: &mut [V], k: usize)
where
    M: SparseMatrix<I, V> + Sync,
    I: Index,
    V: Scalar,
{
    let (m, _) = s.shape();
    let nnz = s.nnz();

    assert_eq!(out.len(), nnz, "Output buffer size mismatch");
    assert!(d1.len() >= m * k, "Dense matrix D1 size mismatch");
    assert!(d2.len() >= k * s.shape().1, "Dense matrix D2 size mismatch");

    s.par_zip_out(out, PartitionStrategy::RowNnzBalance).for_each(
        |(i, col_indices, vals, out_row)| {
            let d1_row = &d1[i * k..(i + 1) * k];
            for (idx, &col_idx) in col_indices.iter().enumerate() {
                let d2_row_start = col_idx.to_usize() * k;
                let d2_row = &d2[d2_row_start..d2_row_start + k];

                let mut dot_product = V::zero();
                for j in 0..k {
                    dot_product += d1_row[j] * d2_row[j];
                }

                out_row[idx] = vals[idx] * dot_product;
            }
        },
    );
}

#[cfg(test)]
mod kernel_tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::core::csr::CsrView;

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
    fn test_sddmm_happy_path() {
        // S (2x3): [[1, 0, 2], [0, 0, 3]]
        // D1 (2x2): [[1, 2], [3, 4]]
        // D2 (3x2): [[5, 6], [7, 8], [9, 10]]
        // Expected output non-zero values:
        // S_00 = 1 * (1*5 + 2*6) = 17
        // S_02 = 2 * (1*9 + 2*10) = 58
        // S_12 = 3 * (3*9 + 4*10) = 141
        let row_ptr = vec![0u32, 2, 3];
        let col_idx = vec![0u32, 2, 2];
        let vals = vec![1.0f32, 2.0, 3.0];
        let s = CsrView::new((2, 3), &row_ptr, &col_idx, &vals);

        let d1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let d2 = vec![5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut out = vec![0.0f32; vals.len()];

        sddmm(&s, &d1, &d2, &mut out, 2);

        assert_eq!(out[0], 17.0); // 1 * (1*5 + 2*6) = 17
        assert_eq!(out[1], 58.0); // 2 * (1*9 + 2*10) = 58
        assert_eq!(out[2], 201.0); // 3 * (3*9 + 4*10) = 201
    }

    #[test]
    fn test_sddmm_empty_row() {
        // S (3x2): [[1, 2], [0, 0], [3, 4]] -> Middle row is empty
        // D1 (3x2): [[1, 2], [3, 4], [5, 6]]
        // D2 (2x2): [[7, 8], [9, 10]]
        let row_ptr = vec![0u32, 2, 2, 4]; // row 1: 2 to 2 (empty)
        let col_idx = vec![0u32, 1, 0, 1];
        let vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let s = CsrView::new((3, 2), &row_ptr, &col_idx, &vals);
        let d1 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let d2 = vec![7.0f32, 8.0, 9.0, 10.0];
        let mut out = vec![0.0f32; vals.len()];

        sddmm(&s, &d1, &d2, &mut out, 2);

        assert_relative_eq!(out[0], 23.0); // 1 * (1*7 + 2*8) = 23
        assert_relative_eq!(out[1], 58.0); // 2 * (1*9 + 2*10) = 58
        assert_relative_eq!(out[2], 249.0); // 3 * (5*7 + 6*8) = 249
        assert_relative_eq!(out[3], 420.0); // 4 * (5*9 + 6*10) = 420
    }
}
