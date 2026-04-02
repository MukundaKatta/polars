use polars_core::prelude::PlHashMap;
use polars_utils::pl_str::PlSmallStr;

use crate::plans::AExprSorted;

pub struct FrameSortColumn {
    pub descending: bool,
    pub nulls_last: bool,
}

pub struct FramePartitioning {
    pub keys: Vec<PlSmallStr>,
    /// keys_sortedness.len() <= keys.len()
    pub keys_sortedness: Vec<FrameSortColumn>,
    pub independent_sorted: PlHashMap<PlSmallStr, AExprSorted>,
}
