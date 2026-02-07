use ndarray::{Array2, array};
use sparsers::{sddmm_ndarray, spmm_ndarray};

fn main() {
    // Example: SpMM using CSR matrix converted from 10 * 10 ndarray
    let sparse_array = array![
        [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];

    let dense_matrix = array![
        // Dense matrix (10x2)
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
        [13.0, 14.0],
        [15.0, 16.0],
        [17.0, 18.0],
        [19.0, 20.0],
    ];
    let result = spmm_ndarray::<u32, f32>(sparse_array.view(), dense_matrix.view());
    println!("Result of SpMM:\n{:#?}", result);

    // Example: SDDMM using CSR matrix converted from 5 * 5 ndarray
    let sparse_array = array![
        [0.0, 0.0, 3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 6.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 7.0],
        [0.0, 0.0, 0.0, 8.0, 0.0],
    ];

    let dense_matrix_d1 = array![
        // Dense matrix d1 (5x1)
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ];
    let dense_matrix_d2 = array![
        // Dense matrix d2 (1x5)
        [1.0, 2.0, 3.0, 4.0, 5.0],
    ];

    let result = sddmm_ndarray::<u32, f32>(
        sparse_array.view(),
        dense_matrix_d1.view(),
        dense_matrix_d2.view(),
    );
    println!("Result of SDDMM:\n{:#?}", result);
}
