use std::fs::File;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarray::{Array1, Array2, ArrayView2};
use ndarray_npy::NpzReader;
use rayon::prelude::*;
use sparsers::core::{common::SparseMatrix, csr::CsrContainer, kernel::spmm_dense};

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

fn bench_spmm_kernel(c: &mut Criterion) {
    let (csr, b_matrix) = setup_roadnet();
    let b_view = b_matrix.view();
    let a_rows = csr.shape().0;
    let b_cols = b_matrix.shape()[1];
    let mut c_output = vec![0.0f64; a_rows * b_cols];

    let mut group: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("Kernel_Performance");
    group.sample_size(10);

    group.bench_function("roadNet-CA-spmm", |b| {
        b.iter(|| {
            spmm_dense(
                &csr,
                b_view.as_slice().unwrap(),
                &mut c_output,
                b_view.shape()[1],
            )
        });
    });

    group.finish();
}

criterion_group!(benches, bench_spmm_kernel);
criterion_main!(benches);
