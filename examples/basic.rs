use ndarray::{Array2, array};
use sparsers::core::{
    common::SparseMatrix,
    csr::CsrContainer,
    kernel::{sddmm_csr, spmm_csr},
};

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

    let sparse_csr: CsrContainer<u32, f32> = CsrContainer::from_ndarray(sparse_array.view());

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
    let dense_slice = dense_matrix.as_slice().unwrap();

    let mut result = vec![0.0f32; 10 * 2]; // Resul buffer for the dense matrix (10x2)

    spmm_csr(
        sparse_csr.view(),
        &dense_slice,
        &mut result,
        dense_matrix.ncols(),
    );
    let res_arr = Array2::from_shape_vec((10, 2), result)
        .expect("Error: Shape mismatch during array conversion");
    println!("Result of SpMM:\n{:#?}", res_arr);

    // Example: SDDMM using CSR matrix converted from 5 * 5 ndarray
    let sparse_array = array![
        [0.0, 0.0, 3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 6.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 7.0],
        [0.0, 0.0, 0.0, 8.0, 0.0],
    ];

    let sparse_csr: CsrContainer<u32, f32> = CsrContainer::from_ndarray(sparse_array.view());

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
    let dense_d1_slice = dense_matrix_d1.as_slice().unwrap();
    let dense_d2_slice = dense_matrix_d2.as_slice().unwrap();

    let mut result = vec![0.0f32; sparse_csr.nnz()]; // Result buffer for the SDDMM

    sddmm_csr(
        sparse_csr.view(),
        dense_d1_slice,
        dense_d2_slice,
        &mut result,
        1,
    );
    println!("Result of SDDMM:\n{:#?}", result);
}
