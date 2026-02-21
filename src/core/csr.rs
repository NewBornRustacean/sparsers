use ndarray::{ArrayView2, Axis, Zip, parallel::prelude::*};
use rayon::prelude::*;

use crate::core::{
    common::{ExecutionPolicy, Index, RowChunkProducer, Scalar, SparseMatrix},
    coo::CooContainer,
};

pub struct CsrContainer<I: Index, V> {
    pub row_ptrs: Vec<I>,
    pub col_indices: Vec<I>,
    pub values: Vec<V>,
    pub shape: (usize, usize),
}

impl<I: Index, V: Scalar> CsrContainer<I, V> {
    #[inline(always)]
    pub fn new(
        shape: (usize, usize),
        row_ptrs: Vec<I>,
        col_indices: Vec<I>,
        values: Vec<V>,
    ) -> Self {
        Self {
            shape,
            row_ptrs,
            col_indices,
            values,
        }
    }

    pub fn from_coo(coo: &CooContainer<I, V>) -> CsrContainer<I, V> {
        let (rows, cols) = coo.shape;
        let nnz = coo.values.len();

        let mut entries: Vec<(I, I, V)> = (0..nnz)
            .into_par_iter()
            .map(|idx| {
                (
                    coo.row_indices[idx],
                    coo.col_indices[idx],
                    coo.values[idx].clone(),
                )
            })
            .collect();

        entries.par_sort_unstable_by_key(|e| e.0.to_usize());

        let mut row_ptrs = vec![I::from_usize(0); rows + 1];
        for &(r, _, _) in &entries {
            let r_idx = r.to_usize();
            row_ptrs[r_idx + 1] = I::from_usize(row_ptrs[r_idx + 1].to_usize() + 1);
        }

        for i in 0..rows {
            let prev = row_ptrs[i].to_usize();
            let curr = row_ptrs[i + 1].to_usize();
            row_ptrs[i + 1] = I::from_usize(prev + curr);
        }

        let (col_indices, values): (Vec<I>, Vec<V>) =
            entries.into_par_iter().map(|(_, c, v)| (c, v)).unzip();

        CsrContainer::new((rows, cols), row_ptrs, col_indices, values)
    }

    /// Create a CSR matrix from an ndarray 2D array
    /// this function bypasses genrating an intermidiate COO matrix.
    /// without extra conversion logics, it directly constructs CSR from ndarray
    pub fn from_ndarray(array: ArrayView2<V>) -> Self {
        let (rows, cols) = array.dim();
        let zero = V::zero();

        let mut row_ptrs = Vec::with_capacity(rows + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        row_ptrs.push(I::from_usize(0));
        for row_view in array.axis_iter(Axis(0)) {
            Zip::indexed(row_view).for_each(|col_idx, &val| {
                if val != zero {
                    col_indices.push(I::from_usize(col_idx));
                    values.push(val);
                }
            });
            row_ptrs.push(I::from_usize(values.len()));
        }

        Self {
            shape: (rows, cols),
            row_ptrs,
            col_indices,
            values,
        }
    }

    pub fn view(&self) -> CsrView<I, V> {
        CsrView::new(self.shape, &self.row_ptrs, &self.col_indices, &self.values)
    }
}

impl<I: Index, V: Scalar> SparseMatrix<I, V> for CsrContainer<I, V> {
    #[inline(always)]
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    #[inline(always)]
    fn nnz(&self) -> usize {
        self.values.len()
    }

    #[inline(always)]
    fn outer_slice(&self, idx: usize) -> Option<(&[I], &[V])> {
        if idx >= self.shape.0 {
            return None;
        }
        let start = self.row_ptrs[idx].to_usize();
        let end = self.row_ptrs[idx + 1].to_usize();
        Some((&self.col_indices[start..end], &self.values[start..end]))
    }

    fn row_offset(&self, idx: usize) -> usize {
        if self.outer_dims() <= idx {
            panic!("Row index out of bounds");
        }
        self.row_ptrs[idx].to_usize()
    }
}

pub struct CsrView<'a, I: Index, V: Scalar> {
    pub row_ptrs: &'a [I],
    pub col_indices: &'a [I],
    pub values: &'a [V],
    pub shape: (usize, usize),
}

impl<'a, I: Index, V: Scalar> CsrView<'a, I, V> {
    #[inline(always)]
    pub fn new(
        shape: (usize, usize),
        row_ptrs: &'a [I],
        col_indices: &'a [I],
        values: &'a [V],
    ) -> Self {
        Self {
            shape,
            row_ptrs,
            col_indices,
            values,
        }
    }

    #[inline(always)]
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    #[inline(always)]
    pub fn row(&self, idx: usize) -> Option<(&[I], &[V])> {
        if idx >= self.shape.0 {
            return None;
        }
        let start = self.row_ptrs[idx].to_usize();
        let end = self.row_ptrs[idx + 1].to_usize();
        Some((&self.col_indices[start..end], &self.values[start..end]))
    }

    pub fn par_zip_out<P: ExecutionPolicy>(
        &'a self,
        out: &'a mut [V],
        policy: P,
    ) -> RowChunkProducer<'a, Self, I, V, P>
    where
        Self: Sized + Sync,
    {
        RowChunkProducer::new(self, out, 0, self.outer_dims(), policy)
    }
}

impl<'a, I: Index, V: Scalar> SparseMatrix<I, V> for CsrView<'a, I, V> {
    #[inline(always)]
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    #[inline(always)]
    fn nnz(&self) -> usize {
        self.values.len()
    }

    #[inline(always)]
    fn outer_slice(&self, idx: usize) -> Option<(&[I], &[V])> {
        if idx >= self.shape.0 {
            return None;
        }
        let start = self.row_ptrs[idx].to_usize();
        let end = self.row_ptrs[idx + 1].to_usize();
        Some((&self.col_indices[start..end], &self.values[start..end]))
    }

    fn row_offset(&self, idx: usize) -> usize {
        self.row_ptrs[idx].to_usize()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_build_csr_from_ndarray_pass() {
        let sparse = array![[0.0, 0.0, 3.0], [4.0, 0.0, 0.0], [0.0, 5.0, 6.0],];

        let csr: CsrContainer<u32, f32> = CsrContainer::from_ndarray(sparse.view());

        assert_eq!(csr.shape, (3, 3));
        assert_eq!(csr.row_ptrs, vec![0u32, 1, 2, 4]);
        assert_eq!(csr.col_indices, vec![2u32, 0, 1, 2]);
        assert_eq!(csr.values, vec![3.0f32, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_build_csr_from_ndarray_empty() {
        let sparse = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],];

        let csr: CsrContainer<u32, f32> = CsrContainer::from_ndarray(sparse.view());

        assert_eq!(csr.shape, (3, 3));
        assert_eq!(csr.row_ptrs, vec![0u32, 0, 0, 0]);
        assert_eq!(csr.col_indices, Vec::<u32>::new());
        assert_eq!(csr.values, Vec::<f32>::new());
    }

    #[test]
    fn test_build_csr_from_ndarray_full() {
        let sparse = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];
        let csr: CsrContainer<u32, f32> = CsrContainer::from_ndarray(sparse.view());
        assert_eq!(csr.shape, (3, 3));
        assert_eq!(csr.row_ptrs, vec![0u32, 3, 6, 9]);
        assert_eq!(csr.col_indices, vec![0u32, 1, 2, 0, 1, 2, 0, 1, 2]);
        assert_eq!(
            csr.values,
            vec![
                1.0f32, 2.0, 3.0, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32
            ]
        );
    }

    #[test]
    fn test_csr_row_access() {
        // 1. a simple 2x3 CSR matrix
        // [1, 0, 2]
        // [0, 0, 3]
        let row_ptrs = vec![0u32, 2, 3];
        let col_indices = vec![0u32, 2, 2];
        let values = vec![1.0f32, 2.0, 3.0];

        let csr = CsrContainer {
            row_ptrs,
            col_indices,
            values,
            shape: (2, 3),
        };

        // 2. verify 1st row: [1, 0, 2] -> 0th, 2nd column have values
        let Some((cols, vals)) = csr.outer_slice(0) else {
            panic!("Row not found")
        };
        assert_eq!(cols, &[0, 2]);
        assert_eq!(vals, &[1.0, 2.0]);

        // 3. verify second row: [0, 0, 3] -> only 2nd column has a value
        let Some((cols, vals)) = csr.outer_slice(1) else {
            panic!("Row not found")
        };
        assert_eq!(cols, &[2]);
        assert_eq!(vals, &[3.0]);
    }

    #[test]
    fn test_empty_row() {
        // A 2x2 CSR matrix with an empty first row
        let row_ptrs = vec![0u32, 0, 0];
        let csr = CsrContainer {
            row_ptrs,
            col_indices: vec![],
            values: vec![] as Vec<f32>,
            shape: (2, 2),
        };

        let Some((cols, vals)) = csr.outer_slice(0) else {
            panic!("Row not found")
        };
        assert!(cols.is_empty());
        assert!(vals.is_empty());
    }

    #[test]
    fn test_csr_row_access_view() {
        // 1. a simple 2x3 CSR matrix
        // [1, 0, 2]
        // [0, 0, 3]
        let row_ptrs = vec![0u32, 2, 3];
        let col_indices = vec![0u32, 2, 2];
        let values = vec![1.0f32, 2.0, 3.0];

        let csr_view = CsrView {
            row_ptrs: &row_ptrs,
            col_indices: &col_indices,
            values: &values,
            shape: (2, 3),
        };

        // 2. verify 1st row: [1, 0, 2] -> 0th, 2nd column have values
        let Some((cols, vals)) = csr_view.row(0) else {
            panic!("Row not found")
        };
        assert_eq!(cols, &[0, 2]);
        assert_eq!(vals, &[1.0, 2.0]);

        // 3. verify second row: [0, 0, 3] -> only 2nd column has a value
        let Some((cols, vals)) = csr_view.row(1) else {
            panic!("Row not found")
        };
        assert_eq!(cols, &[2]);
        assert_eq!(vals, &[3.0]);
    }

    #[test]
    fn test_empty_row_view() {
        // A 2x2 CSR matrix with an empty first row
        let row_ptrs = vec![0u32, 0, 0];
        let csr_view = CsrView {
            row_ptrs: &row_ptrs,
            col_indices: &vec![],
            values: &vec![] as &Vec<f32>,
            shape: (2, 2),
        };

        let Some((cols, vals)) = csr_view.row(0) else {
            panic!("Row not found")
        };
        assert!(cols.is_empty());
        assert!(vals.is_empty());
    }

    #[test]
    fn test_invalid_index_returns_none() {
        let row_ptrs = vec![0u32, 2];
        let csr = CsrContainer {
            row_ptrs,
            col_indices: vec![0, 1],
            values: vec![1.0, 2.0],
            shape: (1, 2),
        };

        // confirm that returns None for out-of-bounds indices
        assert!(csr.outer_slice(1).is_none());
        assert!(csr.outer_slice(100).is_none());

        // valid index returns Some
        assert!(csr.outer_slice(0).is_some());
    }
}
