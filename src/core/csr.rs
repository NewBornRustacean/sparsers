pub trait Index: Copy + Send + Sync + 'static {
    fn to_usize(self) -> usize;
    fn from_usize(value: usize) -> Self;
}

impl Index for u32 {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        value as u32
    }
}

impl Index for u64 {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        value as u64
    }
}

pub struct CsrContainer<I: Index, V> {
    pub row_ptrs: Vec<I>,
    pub col_indices: Vec<I>,
    pub values: Vec<V>,
    pub shape: (usize, usize),
}

pub struct CsrView<'a, I: Index, V> {
    pub row_ptrs: &'a [I],
    pub col_indices: &'a [I],
    pub values: &'a [V],
    pub shape: (usize, usize),
}

pub trait SparseMatrix<I: Index, V> {
    fn shape(&self) -> (usize, usize);
    fn nnz(&self) -> usize;

    // return a specific row as a slice
    fn row(&self, idx: usize) -> Option<(&[I], &[V])>;
    fn rows(&self) -> usize {
        self.shape().0
    }
}

impl<I: Index, V> SparseMatrix<I, V> for CsrContainer<I, V> {
    #[inline(always)]
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    #[inline(always)]
    fn nnz(&self) -> usize {
        self.values.len()
    }

    #[inline(always)]
    fn row(&self, idx: usize) -> Option<(&[I], &[V])> {
        if idx >= self.shape.0 {
            return None;
        }
        let start = self.row_ptrs[idx].to_usize();
        let end = self.row_ptrs[idx + 1].to_usize();
        Some((&self.col_indices[start..end], &self.values[start..end]))
    }
}

impl<'a, I: Index, V> SparseMatrix<I, V> for CsrView<'a, I, V> {
    #[inline(always)]
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    #[inline(always)]
    fn nnz(&self) -> usize {
        self.values.len()
    }

    #[inline(always)]
    fn row(&self, idx: usize) -> Option<(&[I], &[V])> {
        if idx >= self.shape.0 {
            return None;
        }
        let start = self.row_ptrs[idx].to_usize();
        let end = self.row_ptrs[idx + 1].to_usize();
        Some((&self.col_indices[start..end], &self.values[start..end]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let Some((cols, vals)) = csr.row(0) else {
            panic!("Row not found")
        };
        assert_eq!(cols, &[0, 2]);
        assert_eq!(vals, &[1.0, 2.0]);

        // 3. verify second row: [0, 0, 3] -> only 2nd column has a value
        let Some((cols, vals)) = csr.row(1) else {
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

        let Some((cols, vals)) = csr.row(0) else {
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
        assert!(csr.row(1).is_none());
        assert!(csr.row(100).is_none());

        // valid index returns Some
        assert!(csr.row(0).is_some());
    }
}
