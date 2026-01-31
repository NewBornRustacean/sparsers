pub trait Scalar: num_traits::Num + num_traits::NumAssign + Copy + Send + Sync + 'static {}
impl<T> Scalar for T where T: num_traits::Num + num_traits::NumAssign + Copy + Send + Sync + 'static {}

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

pub trait SparseMatrix<I: Index, V> {
    fn shape(&self) -> (usize, usize);
    fn nnz(&self) -> usize;

    // return a specific row as a slice
    fn row(&self, idx: usize) -> Option<(&[I], &[V])>;
    fn rows(&self) -> usize {
        self.shape().0
    }
}

pub trait MutableSparseMatrix<'a, I: Index, V>: SparseMatrix<I, V> {
    fn split_into_rows_mut(self) -> Vec<(usize, &'a [I], &'a mut [V])>;
}
