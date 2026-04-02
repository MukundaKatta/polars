use polars_core::prelude::PlHashMap;
use polars_utils::UnitVec;
use polars_utils::arena::{Arena, Node};
use slotmap::SlotMap;

use crate::plans::IRRandomMethod;
use crate::plans::ir_traversal::{IRNodeEdgeKeys, IRNodeKey, unpack_edges_mut};

pub trait IREdgeProvivder<'a, Edge: 'a> {
    fn unpack_edges_mut<
        const NUM_INPUTS: usize,
        const NUM_OUTPUTS: usize,
        // Workaround for generic_const_exprs, have the caller pass in `NUM_INPUTS + NUM_OUTPUTS`
        const TOTAL_EDGES: usize,
    >(
        &'a mut self,
    ) -> Option<([&'a mut Edge; NUM_INPUTS], [&'a mut Edge; NUM_OUTPUTS])>;

    fn get_in_edge_mut(&'a mut self, idx: usize) -> &'a mut Edge;

    fn get_out_edge_mut(&'a mut self, idx: usize) -> &'a mut Edge;

    fn num_in_edges(&self) -> usize;

    fn num_out_edges(&self) -> usize;
}

pub struct IRTraversalGraphEdgeProvider<'a, EdgeKey: slotmap::Key, Edge>
where
    EdgeKey:,
{
    ir_node_edge_keys: &'a IRNodeEdgeKeys<EdgeKey>,
    edges_map: &'a mut SlotMap<EdgeKey, Edge>,
}

impl<'a, EdgeKey: slotmap::Key, Edge> IREdgeProvivder<'a, Edge>
    for IRTraversalGraphEdgeProvider<'a, EdgeKey, Edge>
{
    fn unpack_edges_mut<
        const NUM_INPUTS: usize,
        const NUM_OUTPUTS: usize,
        // Workaround for generic_const_exprs, have the caller pass in `NUM_INPUTS + NUM_OUTPUTS`
        const TOTAL_EDGES: usize,
    >(
        &'a mut self,
    ) -> Option<([&'a mut Edge; NUM_INPUTS], [&'a mut Edge; NUM_OUTPUTS])> {
        unpack_edges_mut::<EdgeKey, Edge, NUM_INPUTS, NUM_OUTPUTS, TOTAL_EDGES>(
            self.ir_node_edge_keys,
            self.edges_map,
        )
    }

    fn get_in_edge_mut(&'a mut self, idx: usize) -> &'a mut Edge {
        self.edges_map
            .get_mut(self.ir_node_edge_keys.in_edges[idx])
            .unwrap()
    }

    fn get_out_edge_mut(&'a mut self, idx: usize) -> &'a mut Edge {
        self.edges_map
            .get_mut(self.ir_node_edge_keys.out_edges[idx])
            .unwrap()
    }

    fn num_in_edges(&self) -> usize {
        self.ir_node_edge_keys.in_edges.len()
    }

    fn num_out_edges(&self) -> usize {
        self.ir_node_edge_keys.out_edges.len()
    }
}
