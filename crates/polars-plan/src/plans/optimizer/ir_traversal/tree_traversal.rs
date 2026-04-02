use polars_core::prelude::PlHashMap;
use polars_error::PolarsResult;
use polars_utils::arena::{Arena, Node};
use polars_utils::{UnitVec, unitvec};

use crate::plans::ir_traversal::edge_provider::IREdgeProvivder;
use crate::plans::ir_traversal::ir_node_key::IRNodeKey;
use crate::plans::{AExpr, IR};

pub enum VisitState {
    Pre,
    Post,
}

/// Traverse IR trees. Subtrees in caches are visited once.
#[derive(Debug)]
pub struct IRTreeTraversal {
    pub pre_visit: bool,
    pub post_visit: bool,
}

pub struct BasicEdgeProvider<'a, Edge> {
    in_edges: &'a mut [Edge],
    out_edges: &'a mut [Edge],
}

impl IRTreeTraversal {
    pub fn traverse<Visitor, Edge: Default + Clone>(
        &mut self,
        roots: &[Node],
        mut visitor_fn: Visitor,
        ir_arena: &Arena<IR>,
        expr_arena: &Arena<AExpr>,
    ) -> PolarsResult<()>
    where
        Visitor: FnMut(
            VisitState,
            Node,
            &Arena<IR>,
            &Arena<AExpr>,
            &BasicEdgeProvider<Edge>,
        ) -> PolarsResult<()>,
    {
        assert!(self.pre_visit || self.post_visit);

        let mut visited_caches: PlHashMap<IRNodeKey, Edge> = PlHashMap::default();

        for node in roots.iter() {
            traverse_rec(
                self,
                *node,
                &mut visitor_fn,
                ir_arena,
                expr_arena,
                &mut visited_caches,
                None,
            );
        }

        Ok(())
    }
}

fn traverse_rec<Visitor, Edge: Default + Clone>(
    config: &IRTreeTraversal,
    node: Node,
    visitor_fn: &mut Visitor,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    visited_caches: &mut PlHashMap<IRNodeKey, Edge>,
    out_edge: Option<&mut Edge>,
) -> PolarsResult<()>
where
    Visitor: FnMut(
        VisitState,
        Node,
        &Arena<IR>,
        &Arena<AExpr>,
        &BasicEdgeProvider<Edge>,
    ) -> PolarsResult<()>,
{
    let key = IRNodeKey::new(node, ir_arena);

    if let Some(edge) = visited_caches.get(&key) {
        *out_edge.unwrap() = edge.clone();
        return Ok(());
    }

    let ir = ir_arena.get(node);
    let mut in_edges: UnitVec<Edge> = ir.inputs().map(|_| Edge::default()).collect();

    let edge_provider = BasicEdgeProvider {
        in_edges: &mut in_edges,
        out_edges: out_edge.map_or(&mut [], std::slice::from_mut),
    };

    if config.pre_visit {
        visitor_fn(VisitState::Pre, node, ir_arena, expr_arena, &edge_provider)?;
    }

    for (node, edge) in ir.inputs().zip(edge_provider.in_edges.iter_mut()) {
        traverse_rec(
            config,
            node,
            visitor_fn,
            ir_arena,
            expr_arena,
            visited_caches,
            Some(edge),
        )?;
    }

    if config.post_visit {
        visitor_fn(VisitState::Post, node, ir_arena, expr_arena, &edge_provider);
    }

    if matches!(ir_arena.get(node), IR::Cache { .. }) {
        visited_caches.insert(key, todo!());
    }

    Ok(())
}
