use crate::graph::{EdgeData, Graph, NodeId};
use pathfinding::prelude::dijkstra_all;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::rc::Rc;

// An enum for path type for clearer implementation
#[derive(Debug, Clone, Copy)]
pub enum Path {
    Adjusted,
    Intact,
}

/// Convert edge list to adjacency list format to be injested by dijksta_all function via successor
/// Converts f32 weights to u32 by multiplying with 100 and rounding to be used in Dijkstra
/// HashMap<u, Vec<(v, adj_cond/cond)>>
/// u: source, v: destination
impl Graph {
    #[inline]
    fn to_adjacency(&self, kind: Path) -> FxHashMap<NodeId, Vec<(NodeId, u32)>> {
        // First pass: count edges per source node
        let edge_counts = self.count_edges();

        // Initialize the adjacency list with pre-allocated space
        let mut adjacency = FxHashMap::default();
        adjacency.reserve(edge_counts.len());
        for (&node, &count) in &edge_counts {
            adjacency.insert(node, Vec::with_capacity(count));
        }

        // Second pass: fill adjacency list; u32 to avoid integer overflow
        for (&(u, v), edge) in &self.data {
            // This is needed in integers, so multiplied by 100 to get upto 2 digits precision
            let dist = match kind {
                Path::Adjusted => edge.adj_dist,
                Path::Intact => edge.geo_dist,
            };
            let weight = (dist * 100.0).round() as u32;

            if let Some(neighbors) = adjacency.get_mut(&u) {
                neighbors.push((v, weight));
            }
        }

        adjacency
    }
}

pub trait GraphDijkstraExt {
    fn dijkstra(&self, kind: Path) -> HashMap<NodeId, (NodeId, u32)>;
}

impl GraphDijkstraExt for Graph {
    /// Create the reachable path with dijkstra; weighted by condition or not
    fn dijkstra(&self, kind: Path) -> HashMap<NodeId, (NodeId, u32)> {
        let graph_int = self.to_adjacency(kind);
        let successors = |node: &NodeId| graph_int.get(node).into_iter().flatten().copied();

        // Calculate all reachable paths; the end nodes/segments
        dijkstra_all(&self.source, successors)
    }
}

/// Return distance values and the condition/similarity of the last segment
pub fn path_distance(graph: &Graph, path: &[NodeId], dist_intact: f32) -> EdgeData {
    let mut dist_adjusted = 0.0;
    let mut last_condition = 0.0;
    let mut last_pa = 1.0;
    let mut last_num_cells = 0.0;
    let mut last_sims = Rc::new(Vec::new());

    // Loop through nodes in a path to the target node/segment
    for (from, to) in path.windows(2).map(|w| (w[0], w[1])) {
        if let Some(edge) = graph.get(&(from, to)) {
            dist_adjusted += edge.adj_dist;
            last_condition = edge.condition;
            last_pa = edge.pa;
            last_num_cells = edge.num_cells;
            last_sims = Rc::clone(&edge.similarities);
        }
    }

    EdgeData {
        adj_dist: dist_adjusted,
        geo_dist: dist_intact,
        condition: last_condition,
        pa: last_pa,
        num_cells: last_num_cells,
        similarities: last_sims,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_distance_sums_stored_adjusted_edges_without_rescaling() {
        let sims = Rc::new(vec![Some(1.0)]);
        let mut graph = Graph::new(None);
        graph.add_node(
            (0, 1),
            EdgeData {
                adj_dist: 3.0,
                geo_dist: 2.0,
                condition: 0.5,
                pa: 1.0,
                num_cells: 4.0,
                similarities: sims.clone(),
            },
        );
        graph.add_node(
            (1, 2),
            EdgeData {
                adj_dist: 5.0,
                geo_dist: 4.0,
                condition: 0.25,
                pa: 0.5,
                num_cells: 7.0,
                similarities: sims,
            },
        );

        let edge = path_distance(&graph, &[0, 1, 2], 6.0);

        assert!((edge.adj_dist - 8.0).abs() < 1.0e-6);
        assert!((edge.geo_dist - 6.0).abs() < 1.0e-6);
        assert!((edge.condition - 0.25).abs() < 1.0e-6);
        assert!((edge.pa - 0.5).abs() < 1.0e-6);
        assert!((edge.num_cells - 7.0).abs() < 1.0e-6);
        assert_eq!(edge.similarities.as_ref(), &[Some(1.0)]);
    }
}
