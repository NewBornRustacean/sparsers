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

pub trait ExecutionPolicy: Copy + Send + Sync {
    fn split_info<M: SparseMatrix<I, V>, I: Index, V: Scalar>(
        &self,
        matrix: &M,
        start_row: usize,
        num_rows: usize,
        mid: usize,
    ) -> (usize, usize);

    fn chunk_size<I: Index>(&self, cols: &[I]) -> usize;
}

/// partition strategy for spmm: fixed memory layout
#[derive(Copy, Clone, Debug)]
pub struct SpmmPolicy {
    pub k: usize,
}
impl ExecutionPolicy for SpmmPolicy {
    fn split_info<M: SparseMatrix<I, V>, I: Index, V: Scalar>(
        &self,
        _: &M,
        _: usize,
        _: usize,
        mid: usize,
    ) -> (usize, usize) {
        (mid, mid * self.k)
    }

    fn chunk_size<I: Index>(&self, _cols: &[I]) -> usize {
        self.k
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SddmmPolicy;
impl ExecutionPolicy for SddmmPolicy {
    fn split_info<M: SparseMatrix<I, V>, I: Index, V: Scalar>(
        &self,
        matrix: &M,
        start_row: usize,
        num_rows: usize,
        _: usize,
    ) -> (usize, usize) {
        // find the row such that the cumulative nnz up to that row is just above mid
        let start_offset = matrix.row_offset(start_row);
        let end_offset = matrix.row_offset(start_row + num_rows);
        let target_nnz = start_offset + (end_offset - start_offset) / 2;

        let mut low = start_row;
        let mut high = start_row + num_rows;

        while low < high {
            let m = low + (high - low) / 2;
            if matrix.row_offset(m) < target_nnz {
                low = m + 1;
            } else {
                high = m;
            }
        }

        let mid_row_abs = low;
        let mid_low_rel = mid_row_abs - start_row;
        let offset = matrix.row_offset(mid_row_abs) - start_offset;
        (mid_low_rel, offset)
    }

    fn chunk_size<I: Index>(&self, cols: &[I]) -> usize {
        cols.len()
    }
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
    fn par_zip_out<'a, P: ExecutionPolicy>(
        &'a self,
        out: &'a mut [V],
        policy: P,
    ) -> RowChunkProducer<'a, Self, I, V, P>
    where
        Self: Sized + Sync,
    {
        RowChunkProducer::new(self, out, 0, self.rows(), policy)
    }
}

pub struct RowChunkProducer<'a, M, I, V, P>
where
    M: SparseMatrix<I, V> + Sync,
    I: Index,
    V: Scalar,
    P: ExecutionPolicy,
{
    pub matrix: &'a M,
    pub out: &'a mut [V], // mutable output buffer slice
    pub start_row: usize,
    pub num_rows: usize,
    pub policy: P,
    pub _marker: PhantomData<I>,
}

impl<'a, M, I, V, P> RowChunkProducer<'a, M, I, V, P>
where
    M: SparseMatrix<I, V> + Sync,
    I: Index,
    V: Scalar,
    P: ExecutionPolicy,
{
    pub fn new(
        matrix: &'a M,
        out: &'a mut [V],
        start_row: usize,
        num_rows: usize,
        policy: P,
    ) -> Self {
        Self {
            matrix,
            out,
            start_row,
            num_rows,
            policy,
            _marker: PhantomData,
        }
    }
}

impl<'a, M, I, V, P> Producer for RowChunkProducer<'a, M, I, V, P>
where
    M: SparseMatrix<I, V> + Sync,
    I: Index,
    V: Scalar,
    P: ExecutionPolicy,
{
    type Item = (usize, &'a [I], &'a [V], &'a mut [V]);
    type IntoIter = RowChunkWorker<'a, M, I, V, P>;

    fn into_iter(self) -> Self::IntoIter {
        RowChunkWorker {
            matrix: self.matrix,
            out: self.out,
            start_row: self.start_row,
            num_rows: self.num_rows,
            policy: self.policy,
            _marker: PhantomData,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (mid_row_rel, split_offset) =
            self.policy.split_info(self.matrix, self.start_row, self.num_rows, index);
        let (left_out, right_out) = self.out.split_at_mut(split_offset);
        (
            RowChunkProducer::new(
                self.matrix,
                left_out,
                self.start_row,
                mid_row_rel,
                self.policy,
            ),
            RowChunkProducer::new(
                self.matrix,
                right_out,
                self.start_row + mid_row_rel,
                self.num_rows - mid_row_rel,
                self.policy,
            ),
        )
    }
}

pub struct RowChunkWorker<'a, M, I, V, P>
where
    M: SparseMatrix<I, V> + Sync,
    I: Index,
    V: Scalar,
    P: ExecutionPolicy,
{
    pub matrix: &'a M,
    pub out: &'a mut [V], // mutable output buffer slice
    pub start_row: usize,
    pub num_rows: usize,
    pub policy: P,
    pub _marker: PhantomData<I>,
}

impl<'a, M, I, V, P> Iterator for RowChunkWorker<'a, M, I, V, P>
where
    M: SparseMatrix<I, V> + Sync,
    I: Index,
    V: Scalar,
    P: ExecutionPolicy,
{
    type Item = (usize, &'a [I], &'a [V], &'a mut [V]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.num_rows == 0 {
            return None;
        }

        let curr_row = self.start_row;
        let (cols, vals) = self.matrix.row(curr_row)?;
        let chunk_size = self.policy.chunk_size(cols);

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

impl<'a, M, I, V, P> DoubleEndedIterator for RowChunkWorker<'a, M, I, V, P>
where
    M: SparseMatrix<I, V> + Sync,
    I: Index,
    V: Scalar,
    P: ExecutionPolicy,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.num_rows == 0 {
            return None;
        }

        let last_idx = self.start_row + self.num_rows - 1;
        let (cols, vals) = self.matrix.row(last_idx)?;
        let chunk_size = self.policy.chunk_size(cols);

        let full_out = std::mem::take(&mut self.out);
        let split_point = full_out.len() - chunk_size;
        let (remaining, row_out) = full_out.split_at_mut(split_point);

        self.out = remaining;
        self.num_rows -= 1;

        Some((last_idx, cols, vals, row_out))
    }
}

impl<'a, M, I, V, P> ExactSizeIterator for RowChunkWorker<'a, M, I, V, P>
where
    M: SparseMatrix<I, V> + Sync,
    I: Index,
    V: Scalar,
    P: ExecutionPolicy,
{
    fn len(&self) -> usize {
        self.num_rows
    }
}

impl<'a, M, I, V, P> ParallelIterator for RowChunkProducer<'a, M, I, V, P>
where
    M: SparseMatrix<I, V> + Sync,
    I: Index,
    V: Scalar,
    P: ExecutionPolicy,
{
    type Item = (usize, &'a [I], &'a [V], &'a mut [V]);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.num_rows)
    }
}

impl<'a, M, I, V, P> IndexedParallelIterator for RowChunkProducer<'a, M, I, V, P>
where
    M: SparseMatrix<I, V> + Sync,
    I: Index,
    V: Scalar,
    P: ExecutionPolicy,
{
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.num_rows
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(self)
    }
}
