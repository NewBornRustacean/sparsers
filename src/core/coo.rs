use ndarray::Array2;

use crate::core::common::{Index, Scalar};

pub struct CooContainer<I: Index, V: Scalar> {
    pub row_indices: Vec<I>,
    pub col_indices: Vec<I>,
    pub values: Vec<V>,
    pub shape: (usize, usize),
}

impl<I: Index, V: Scalar> CooContainer<I, V> {
    #[inline(always)]
    pub fn new(
        row_indices: Vec<I>,
        col_indices: Vec<I>,
        values: Vec<V>,
        shape: (usize, usize),
    ) -> Self {
        Self {
            row_indices,
            col_indices,
            values,
            shape,
        }
    }

    #[inline(always)]
    pub fn from_ndarray(array: &Array2<V>) -> Self {
        let shape = array.dim();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for ((i, j), &value) in array.indexed_iter() {
            if value != V::zero() {
                row_indices.push(I::from_usize(i));
                col_indices.push(I::from_usize(j));
                values.push(value);
            }
        }

        Self {
            row_indices,
            col_indices,
            values,
            shape,
        }
    }
}

mod tests {
    use ndarray::array;

    #[allow(unused)]
    use super::*;

    #[test]
    fn test_coo_from_ndarray() {
        let array = array![[0, 0, 3], [4, 0, 0], [0, 5, 6]];
        let coo: CooContainer<u32, i32> = CooContainer::from_ndarray(&array);

        assert_eq!(coo.row_indices, vec![0, 1, 2, 2]);
        assert_eq!(coo.col_indices, vec![2, 0, 1, 2]);
        assert_eq!(coo.values, vec![3, 4, 5, 6]);
        assert_eq!(coo.shape, (3, 3));
    }

    #[test]
    fn test_all_zero_matrix() {
        let array: Array2<f32> = Array2::zeros((4, 4));
        let coo: CooContainer<u32, f32> = CooContainer::from_ndarray(&array);

        assert_eq!(coo.values.len(), 0);
        assert_eq!(coo.row_indices.len(), 0);
        assert_eq!(coo.shape, (4, 4));
    }

    #[test]
    fn test_single_element_at_boundary() {
        // non zero value in the last loc of (2, 2) matrix
        let array = array![[0, 0], [0, 99]];
        let coo: CooContainer<u32, i32> = CooContainer::from_ndarray(&array);

        assert_eq!(coo.values, vec![99]);
        assert_eq!(coo.row_indices, vec![1]);
        assert_eq!(coo.col_indices, vec![1]);
    }

    #[test]
    fn test_non_square_matrix() {
        // 3x2 Matrix
        let array = array![[1, 0], [0, 0], [0, 2]];
        let coo: CooContainer<u32, i32> = CooContainer::from_ndarray(&array);

        assert_eq!(coo.shape, (3, 2));
        assert_eq!(coo.values, vec![1, 2]);
        assert_eq!(coo.row_indices, vec![0, 2]);
        assert_eq!(coo.col_indices, vec![0, 1]);
    }
}
