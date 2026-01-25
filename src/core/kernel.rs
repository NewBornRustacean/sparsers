use rayon::prelude::*;

use crate::csr::{CsrContainer, CsrView, Index};

pub trait Scalar: num_traits::Num + num_traits::NumAssign + Copy + Send + Sync + 'static {}
impl<T> Scalar for T where T: num_traits::Num + num_traits::NumAssign + Copy + Send + Sync + 'static {}

pub fn spmm_dense<I: Index, V: Scalar>(a: &CsrView<I, V>, b: &[V], c: &mut [V], b_cols: usize) {
    let (a_rows, a_cols) = a.shape();
    assert_eq!(c.len(), a_rows * b_cols, "Output buffer size mismatch");
    assert!(b.len() >= a_cols * b_cols, "Dense matrix B is too small");

    c.fill(V::zero());

    c.par_chunks_mut(b_cols).enumerate().for_each(|(i, c_row)| {
        if let Some((a_col_indices, a_values)) = a.row(i) {
            for (&col_index, &val) in a_col_indices.iter().zip(a_values.iter()) {
                let b_row_start = col_index.to_usize() * b_cols;
                let b_row = &b[b_row_start..b_row_start + b_cols];

                for (c_out, &b_val) in c_row.iter_mut().zip(b_row.iter()) {
                    *c_out += val * b_val;
                }
            }
        }
    });
}

#[cfg(test)]
mod kernel_tests {
    use super::*;

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
}
