use std::fs::File;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarray::{Array1, Array2, ArrayView2};
use ndarray_npy::NpzReader;
use rayon::prelude::*;
use sparsers::core::{common::SparseMatrix, csr::CsrContainer, kernel::spmm_csr};

fn setup_roadnet() -> (CsrContainer<i32, f64>, Array2<f64>) {
    let npy_path = "data/roadNet-CA.csr.npz".to_string();
    let mut npz = NpzReader::new(File::open(npy_path).unwrap()).unwrap();

    let indptr: Array1<i32> = npz.by_name("indptr.npy").expect("Missing indptr");
    let indices: Array1<i32> = npz.by_name("indices.npy").expect("Missing indices");
    let data: Array1<f64> = npz.by_name("data.npy").expect("Missing data");
    let shape: Array1<i64> = npz.by_name("shape.npy").expect("Missing shape");

    let csr = CsrContainer::new(
        (shape[0] as usize, shape[1] as usize),
        indptr.to_vec(),
        indices.to_vec(),
        data.to_vec(),
    );
    let feature_dim = 128;
    let b_matrix = Array2::from_elem((shape[1] as usize, feature_dim), 1.0f64);

    (csr, b_matrix)
}

fn spmm_ndarray_parallel(
    row_ptrs: &[i32],
    col_indices: &[i32],
    values: &[f64],
    b: &Array2<f64>,
    c: &mut [f64],
) {
    let b_cols = b.ncols();

    c.par_chunks_mut(b_cols).enumerate().for_each(|(i, row_out)| {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;

        for val in row_out.iter_mut() {
            *val = 0.0;
        }

        for idx in start..end {
            let col = col_indices[idx] as usize;
            let val = values[idx];

            let b_row = b.row(col);
            for j in 0..b_cols {
                row_out[j] += val * b_row[j];
            }
        }
    });
}

fn bench_spmm_comparison(c: &mut Criterion) {
    let (csr, b_matrix) = setup_roadnet();
    let b_view = b_matrix.view();
    let a_rows = csr.shape().0;
    let b_cols = b_matrix.shape()[1];
    let mut c_output = vec![0.0f64; a_rows * b_cols];

    let mut group = c.benchmark_group("SpMM_Comparison");
    group.sample_size(10);

    // 1. sparsers (Our Optimized Kernel)
    group.bench_function("sparsers-optimized", |b| {
        b.iter(|| {
            spmm_csr(
                csr.view(),
                b_view.as_slice().unwrap(),
                &mut c_output,
                b_cols,
            )
        });
    });

    // 2. ndarray parallel loop (Baseline)
    group.bench_function("ndarray-parallel-loop", |b| {
        b.iter(|| {
            spmm_ndarray_parallel(
                &csr.row_ptrs,
                &csr.col_indices,
                &csr.values,
                &b_matrix,
                &mut c_output,
            )
        });
    });

    group.finish();
}

criterion_group!(benches, bench_spmm_comparison);
criterion_main!(benches);
