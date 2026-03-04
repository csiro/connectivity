use std::rc::Rc;
use std::collections::HashMap;
use pathfinding::prelude::dijkstra_all;
use crate::graph::{Graph, EdgeData, NodeId};


// An enum for path type for clearer implementation
#[derive(Debug, Clone, Copy)]
pub enum Path {
    Adjusted,
    Intact
}


/// Convert edge list to adjacency list format to be injested by dijksta_all function via successor
/// Converts f32 weights to u32 by multiplying with 100 and rounding to be used in Dijkstra
/// HashMap<u, Vec<(v, adj_cond/cond)>>
/// u: source, v: destination
impl Graph {
    #[inline]
    fn to_adjacency(
        &self,
        kind: Path,
    ) -> HashMap<NodeId, Vec<(NodeId, u32)>> {
        // First pass: count edges per source node
        let edge_counts = self.count_edges();
    
        // Initialize the adjacency list with pre-allocated space
        let mut adjacency: HashMap<NodeId, Vec<(NodeId, u32)>> = HashMap::with_capacity(edge_counts.len());
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
        let successors = |node: &NodeId| -> Vec<(NodeId, u32)> {
            graph_int.get(node).cloned().unwrap_or_default()
        };
    
        // Calculate all reachable paths; the end nodes/segments
        dijkstra_all(&self.source, successors)
    }
}


/// Return distance values and the condition/similarity of the last segment
pub fn path_distance(
    graph: &Graph,
    path: &[NodeId],
    dist_intact: f32
) -> EdgeData {
    let mut dist_adjusted = 0.0;
    let mut last_condition = 0.0;
    let mut last_num_cells = 0.0;
    let mut last_sims = Rc::new(Vec::new());

    // Loop through nodes in a path to the target node/segment
    for (from, to) in path.windows(2).map(|w| (w[0], w[1])) {
        if let Some(edge) = graph.get(&(from, to)) {
            dist_adjusted += edge.geo_dist / (0.5 * edge.condition + 0.5);
            last_condition = edge.condition;
            last_num_cells = edge.num_cells;
            last_sims = Rc::clone(&edge.similarities);
        }
    }

    EdgeData {
        adj_dist: dist_adjusted,
        geo_dist: dist_intact,
        condition: last_condition,
        num_cells: last_num_cells,
        similarities: last_sims,
    }
}
