use polars_core::prelude::PlHashMap;
use polars_error::PolarsResult;
use polars_utils::arena::{Arena, Node};
use polars_utils::{UnitVec, unitvec};

use crate::plans::ir_traversal::edge_provider::EdgesProvider;
use crate::plans::ir_traversal::ir_node_key::IRNodeKey;
use crate::plans::{AExpr, IR};

pub struct BasicEdgeProvider<'a, Edge> {
    in_edges: &'a mut [Edge],
    out_edges: &'a mut [Edge],
}

impl<'provider, Edge> EdgesProvider<Edge> for BasicEdgeProvider<'provider, Edge> {
    fn unpack_edges_mut<
        'a,
        const NUM_INPUTS: usize,
        const NUM_OUTPUTS: usize,
        const TOTAL_EDGES: usize,
    >(
        &'a mut self,
    ) -> Option<([&'a mut Edge; NUM_INPUTS], [&'a mut Edge; NUM_OUTPUTS])>
    where
        Edge: 'a,
    {
        const {
            assert!(NUM_INPUTS + NUM_OUTPUTS == TOTAL_EDGES);
        }

        Some((
            self.in_edges
                .get_disjoint_mut(std::array::from_fn(|i| i))
                .unwrap(),
            self.out_edges
                .get_disjoint_mut(std::array::from_fn(|i| i))
                .unwrap(),
        ))
    }

    fn num_in_edges(&self) -> usize {
        self.in_edges.len()
    }

    fn num_out_edges(&self) -> usize {
        self.out_edges.len()
    }

    fn get_in_edge_mut<'a>(&'a mut self, idx: usize) -> &'a mut Edge
    where
        Edge: 'a,
    {
        &mut self.in_edges[idx]
    }

    fn get_out_edge_mut<'a>(&'a mut self, idx: usize) -> &'a mut Edge
    where
        Edge: 'a,
    {
        &mut self.out_edges[idx]
    }
}

#[recursive::recursive]
pub fn ir_pullup_traversal_rec<Visitor, Edge: Default + Clone>(
    node: Node,
    visitor_fn: &mut Visitor,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    cache: &mut PlHashMap<IRNodeKey, Edge>,
    cache_all: bool,
) -> PolarsResult<Edge>
where
    Visitor:
        FnMut(Node, &Arena<IR>, &Arena<AExpr>, &mut BasicEdgeProvider<Edge>) -> PolarsResult<()>,
{
    let key = IRNodeKey::new(node, ir_arena);

    if let Some(edge) = cache.get(&key) {
        return Ok(edge.clone());
    }

    let ir = ir_arena.get(node);
    let mut in_edges: UnitVec<Edge> = ir.inputs().map(|_| Edge::default()).collect();

    let mut out_edge = Edge::default();

    let mut edge_provider = BasicEdgeProvider {
        in_edges: &mut in_edges,
        out_edges: std::slice::from_mut(&mut out_edge),
    };

    for (node, edge) in ir.inputs().zip(edge_provider.in_edges.iter_mut()) {
        ir_pullup_traversal_rec(node, visitor_fn, ir_arena, expr_arena, cache, cache_all)?;
    }

    visitor_fn(node, ir_arena, expr_arena, &mut edge_provider);

    if matches!(ir_arena.get(node), IR::Cache { .. }) || cache_all {
        let existing = cache.insert(key, out_edge.clone());
        assert!(existing.is_none() || !matches!(ir_arena.get(node), IR::Cache { .. }));
    }

    Ok(out_edge)
}
