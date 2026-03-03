use std::rc::Rc;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use crate::graph::{Graph, EdgeData, NodeId};


// An enum for path type for clearer implementation
#[derive(Debug, Clone, Copy)]
pub enum Path {
    Adjusted,
    Intact
}

/// Integer scaling used to quantize floating path costs for Dijkstra.
/// Higher precision reduces accidental equal-cost ties.
pub const PATH_WEIGHT_SCALE: f32 = 1_000_000.0;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct State {
    cost: u64,
    node: NodeId,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap semantics.
        other
            .cost
            .cmp(&self.cost)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
fn quantize_dist(dist: f32) -> u64 {
    if !dist.is_finite() || dist <= 0.0 {
        return 0;
    }

    let scaled = (dist as f64) * (PATH_WEIGHT_SCALE as f64);
    if scaled >= (u64::MAX as f64) {
        u64::MAX
    } else {
        scaled.round() as u64
    }
}


/// Convert edge list to adjacency list format for deterministic shortest-path search.
/// Converts f32 weights to u64 by high-precision quantization.
/// HashMap<u, Vec<(v, adj_cond/cond)>>
/// u: source, v: destination
impl Graph {
    #[inline]
    fn to_adjacency(
        &self,
        kind: Path,
    ) -> HashMap<NodeId, Vec<(NodeId, u64)>> {
        // First pass: count edges per source node
        let edge_counts = self.count_edges();
    
        // Initialize the adjacency list with pre-allocated space
        let mut adjacency: HashMap<NodeId, Vec<(NodeId, u64)>> = HashMap::with_capacity(edge_counts.len());
        for (&node, &count) in &edge_counts {
            adjacency.insert(node, Vec::with_capacity(count));
        }
    
        // Second pass: fill adjacency list with quantized distances.
        for (&(u, v), edge) in &self.data {
            let dist = match kind {
                Path::Adjusted => edge.adj_dist,
                Path::Intact => edge.geo_dist,
            };
            let weight = quantize_dist(dist);
    
            if let Some(neighbors) = adjacency.get_mut(&u) {
                neighbors.push((v, weight));
            }
        }

        // Make successor traversal deterministic across runs/tiles.
        // This avoids tie-breaking differences in Dijkstra caused by HashMap iteration order.
        for neighbors in adjacency.values_mut() {
            neighbors.sort_unstable_by_key(|&(v, w)| (v, w));
        }
    
        adjacency
    }
}


pub trait GraphDijkstraExt {
    fn dijkstra(&self, kind: Path) -> HashMap<NodeId, (NodeId, u64)>;
}

impl GraphDijkstraExt for Graph {
    /// Deterministic Dijkstra with explicit equal-cost tie-break on predecessor node id.
    fn dijkstra(&self, kind: Path) -> HashMap<NodeId, (NodeId, u64)> {
        let adjacency = self.to_adjacency(kind);
        let mut dist: HashMap<NodeId, u64> = HashMap::with_capacity(adjacency.len() + 1);
        let mut prev: HashMap<NodeId, NodeId> = HashMap::with_capacity(adjacency.len());
        let mut heap: BinaryHeap<State> = BinaryHeap::new();

        dist.insert(self.source, 0);
        heap.push(State {
            cost: 0,
            node: self.source,
        });

        while let Some(State { cost, node }) = heap.pop() {
            let best = *dist.get(&node).unwrap_or(&u64::MAX);
            if cost > best {
                continue;
            }

            if let Some(neighbors) = adjacency.get(&node) {
                for &(next_node, edge_cost) in neighbors {
                    let next_cost = cost.saturating_add(edge_cost);

                    match dist.get(&next_node).copied() {
                        None => {
                            dist.insert(next_node, next_cost);
                            prev.insert(next_node, node);
                            heap.push(State {
                                cost: next_cost,
                                node: next_node,
                            });
                        }
                        Some(curr_cost) if next_cost < curr_cost => {
                            dist.insert(next_node, next_cost);
                            prev.insert(next_node, node);
                            heap.push(State {
                                cost: next_cost,
                                node: next_node,
                            });
                        }
                        Some(curr_cost) if next_cost == curr_cost => {
                            // Deterministic tie-break: keep lexicographically smaller predecessor.
                            let old_prev = prev.get(&next_node).copied().unwrap_or(u64::MAX);
                            if node < old_prev {
                                prev.insert(next_node, node);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        let mut result: HashMap<NodeId, (NodeId, u64)> = HashMap::with_capacity(prev.len());
        for (node, predecessor) in prev {
            if let Some(&cost) = dist.get(&node) {
                result.insert(node, (predecessor, cost));
            }
        }

        result
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
