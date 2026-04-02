use polars_core::prelude::PlHashMap;
use polars_utils::UnitVec;
use polars_utils::arena::{Arena, Node};
use slotmap::SlotMap;

use crate::plans::IRRandomMethod;
use crate::plans::ir_traversal::{IRNodeEdgeKeys, IRNodeKey, unpack_edges_mut};

pub trait IREdgeProvivder<Edge> {
    fn unpack_edges_mut<
        'a,
        const NUM_INPUTS: usize,
        const NUM_OUTPUTS: usize,
        // Workaround for generic_const_exprs, have the caller pass in `NUM_INPUTS + NUM_OUTPUTS`
        const TOTAL_EDGES: usize,
    >(
        &'a mut self,
    ) -> Option<([&'a mut Edge; NUM_INPUTS], [&'a mut Edge; NUM_OUTPUTS])>
    where
        Edge: 'a;

    fn get_in_edge_mut<'a>(&'a mut self, idx: usize) -> &'a mut Edge
    where
        Edge: 'a;

    fn get_out_edge_mut<'a>(&'a mut self, idx: usize) -> &'a mut Edge
    where
        Edge: 'a;

    fn num_in_edges(&self) -> usize;

    fn num_out_edges(&self) -> usize;

    fn map_in_edges_mut<'a, O, F>(&'a mut self, mut f: F) -> impl Iterator<Item = O> + 'a
    where
        Edge: 'a,
        F: for<'b> FnMut(&'b mut Edge) -> O + 'a,
    {
        (0..self.num_in_edges()).map(move |i| f(self.get_in_edge_mut(i)))
    }

    fn map_out_edges_mut<'a, O, F>(&'a mut self, mut f: F) -> impl Iterator<Item = O>
    where
        Edge: 'a,
        F: for<'b> FnMut(&'b mut Edge) -> O + 'a,
    {
        (0..self.num_out_edges()).map(move |i| f(self.get_out_edge_mut(i)))
    }
}

pub struct IRTraversalGraphEdgeProvider<'a, EdgeKey: slotmap::Key, Edge>
where
    EdgeKey:,
{
    pub ir_node_edge_keys: &'a IRNodeEdgeKeys<EdgeKey>,
    pub edges_map: &'a mut SlotMap<EdgeKey, Edge>,
}

impl<'provider, EdgeKey: slotmap::Key, Edge> IREdgeProvivder<Edge>
    for IRTraversalGraphEdgeProvider<'provider, EdgeKey, Edge>
{
    fn unpack_edges_mut<
        'a,
        const NUM_INPUTS: usize,
        const NUM_OUTPUTS: usize,
        // Workaround for generic_const_exprs, have the caller pass in `NUM_INPUTS + NUM_OUTPUTS`
        const TOTAL_EDGES: usize,
    >(
        &'a mut self,
    ) -> Option<([&'a mut Edge; NUM_INPUTS], [&'a mut Edge; NUM_OUTPUTS])>
    where
        Edge: 'a,
    {
        unpack_edges_mut::<EdgeKey, Edge, NUM_INPUTS, NUM_OUTPUTS, TOTAL_EDGES>(
            self.ir_node_edge_keys,
            self.edges_map,
        )
    }

    fn get_in_edge_mut<'a>(&'a mut self, idx: usize) -> &'a mut Edge
    where
        Edge: 'a,
    {
        self.edges_map
            .get_mut(self.ir_node_edge_keys.in_edges[idx])
            .unwrap()
    }

    fn get_out_edge_mut<'a>(&'a mut self, idx: usize) -> &'a mut Edge
    where
        Edge: 'a,
    {
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
