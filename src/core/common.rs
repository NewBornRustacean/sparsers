use std::marker::PhantomData;

use rayon::{iter::plumbing::*, prelude::*};

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

impl Index for i32 {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        value as i32
    }
}

impl Index for i64 {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        value as i64
    }
}

#[derive(Copy, Clone, Debug)]
pub enum PartitionStrategy {
    Fixed(usize),
    RowNnzBalance,
}

pub trait SparseMatrix<I: Index, V: Scalar> {
    fn shape(&self) -> (usize, usize);
    fn nnz(&self) -> usize;

    // return a specific row as a slice
    fn row(&self, idx: usize) -> Option<(&[I], &[V])>;
    fn rows(&self) -> usize {
        self.shape().0
    }

    // return the starting offset of a specific row
    fn row_offset(&self, idx: usize) -> usize;

    // parallel row iterator with output buffer
    fn par_zip_out<'a>(
        &'a self,
        out: &'a mut [V],
        strategy: PartitionStrategy,
    ) -> RowChunkProducer<'a, Self, I, V>
    where
        Self: Sized + Sync,
    {
        RowChunkProducer::new(self, out, 0, self.rows(), strategy)
    }
}

pub struct RowChunkProducer<'a, M, I, V>
where
    M: SparseMatrix<I, V> + Sync,
    I: Index,
    V: Scalar,
{
    pub matrix: &'a M,
    pub out: &'a mut [V], // mutable output buffer slice
    pub start_row: usize,
    pub num_rows: usize,
    pub strategy: PartitionStrategy,
    pub _marker: PhantomData<I>,
}

impl<'a, M, I, V> RowChunkProducer<'a, M, I, V>
where
    M: SparseMatrix<I, V> + Sync,
    I: Index,
    V: Scalar,
{
    pub fn new(
        matrix: &'a M,
        out: &'a mut [V],
        start_row: usize,
        num_rows: usize,
        strategy: PartitionStrategy,
    ) -> Self {
        Self {
            matrix,
            out,
            start_row,
            num_rows,
            strategy,
            _marker: PhantomData,
        }
    }
}

// Sequential Iterator
impl<'a, M, I, V> Iterator for RowChunkProducer<'a, M, I, V>
where
    M: SparseMatrix<I, V> + Sync,
    I: Index,
    V: Scalar,
{
    type Item = (usize, &'a [I], &'a [V], &'a mut [V]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.num_rows == 0 {
            return None;
        }

        let curr_row = self.start_row;
        let (cols, vals) = self.matrix.row(curr_row)?;

        let chunk_size = match self.strategy {
            PartitionStrategy::RowNnzBalance => cols.len(),
            PartitionStrategy::Fixed(k) => k,
        };

        let full_out = std::mem::take(&mut self.out);
        let (row_out, remaining) = full_out.split_at_mut(chunk_size);
        self.out = remaining;

        self.start_row += 1;
        self.num_rows -= 1;

        Some((curr_row, cols, vals, row_out))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.num_rows, Some(self.num_rows))
    }
}

impl<'a, M, I, V> ExactSizeIterator for RowChunkProducer<'a, M, I, V>
where
    M: SparseMatrix<I, V> + Sync,
    I: Index,
    V: Scalar,
{
    fn len(&self) -> usize {
        self.num_rows
    }
}

impl<'a, M, I, V> DoubleEndedIterator for RowChunkProducer<'a, M, I, V>
where
    M: SparseMatrix<I, V> + Sync,
    I: Index,
    V: Scalar,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.num_rows == 0 {
            return None;
        }
        let last_idx = self.start_row + self.num_rows - 1;
        let (cols, vals) = self.matrix.row(last_idx)?;

        let chunk_size = match self.strategy {
            PartitionStrategy::RowNnzBalance => cols.len(),
            PartitionStrategy::Fixed(k) => k,
        };

        let full_out = std::mem::take(&mut self.out);
        let split_point = full_out.len() - chunk_size;

        let (remaining, row_out) = full_out.split_at_mut(split_point);

        self.out = remaining;
        self.num_rows -= 1;

        Some((last_idx, cols, vals, row_out))
    }
}
// parallel producer
impl<'a, M, I, V> Producer for RowChunkProducer<'a, M, I, V>
where
    M: SparseMatrix<I, V> + Sync,
    I: Index,
    V: Scalar,
{
    type Item = (usize, &'a [I], &'a [V], &'a mut [V]);
    type IntoIter = Self;

    fn into_iter(self) -> Self::IntoIter {
        self
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mid_row = index;

        let split_offset = match self.strategy {
            PartitionStrategy::RowNnzBalance => {
                let abs_mid = self.start_row + mid_row;
                self.matrix.row_offset(abs_mid) - self.matrix.row_offset(self.start_row)
            }
            PartitionStrategy::Fixed(k) => mid_row * k,
        };

        let (left_out, right_out) = self.out.split_at_mut(split_offset);

        (
            RowChunkProducer {
                matrix: self.matrix,
                out: left_out,
                start_row: self.start_row,
                num_rows: mid_row,
                strategy: self.strategy,
                _marker: PhantomData,
            },
            RowChunkProducer {
                matrix: self.matrix,
                out: right_out,
                start_row: self.start_row + mid_row,
                num_rows: self.num_rows - mid_row,
                strategy: self.strategy,
                _marker: PhantomData,
            },
        )
    }
}
