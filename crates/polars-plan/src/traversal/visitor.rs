use std::ops::ControlFlow;

use polars_error::PolarsResult;

use crate::traversal::edge_provider::NodeEdgesProvider;

pub enum SubtreeVisit {
    Visit,
    Skip,
}

pub trait NodeVisitor {
    type Key;
    type Storage;
    type Edge;

    fn default_edge(&mut self) -> Self::Edge;

    fn pre_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> PolarsResult<ControlFlow<(), SubtreeVisit>>;

    fn post_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> PolarsResult<ControlFlow<(), ()>>;
}
