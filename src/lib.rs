pub mod core;
pub mod interface;

pub use crate::{
    core::common::{Index, Scalar},
    interface::{
        array::{sddmm_ndarray, spmm_ndarray},
        error::{SparsersError, SparsersResult},
    },
};
