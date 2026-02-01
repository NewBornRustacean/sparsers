use ndarray::{Array2, array};
use sparsers::core::{
    coo::CooContainer,
    csr::{CsrContainer, CsrView, CsrViewMut},
    kernel::{sddmm, spmm_dense},
};

fn main() {
    // 1. ndarray. 10 * 10 sparse matrix with 4 non-zero elements
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

    // 2. Convert ndarray to CSR format
    let sparse_csr: CsrContainer<u32, f32> = CsrContainer::from_ndarray(&sparse_array);

    // 3. example spmm_dense with 10 *2 dense matrix
    let dense_matrix = array![
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

    spmm_dense(&sparse_csr, &dense_slice, &mut result, dense_matrix.ncols());
    let res_arr = Array2::from_shape_vec((10, 2), result)
        .expect("Error: Shape mismatch during array conversion");
    println!("Result of SpMM:\n{:#?}", res_arr);
}
