//! Frozen baseline routing (95554da `routing.rs`), kept for tests.
use super::graph::NodeId;
use super::graph::{EdgeData, Graph};
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::rc::Rc;

/// Compact adjacency: offsets delimit each node's outgoing edges in one flat array.
/// Edges hold (destination, adjusted cost, intact cost), with matching edge data.
pub struct Adjacency<'a> {
    pub offsets: Vec<usize>,
    pub edges: Vec<(usize, u32, u32)>,
    pub data: Vec<&'a EdgeData>,
    #[cfg(test)]
    pub nodes: Vec<NodeId>,
}

/// Convert packed node IDs to local indices once, before either shortest-path search.
impl Graph {
    #[inline]
    pub fn to_adjacency(&self) -> Adjacency<'_> {
        let mut index = FxHashMap::default();
        index.insert(self.source, 0);
        let mut counts = vec![0usize];
        let mut endpoints = Vec::with_capacity(self.data.len());

        // Retain the edge-table order: it determines parents when integer costs tie.
        for &(u, v) in self.data.keys() {
            let next = index.len();
            let from = *index.entry(u).or_insert(next);
            let next = index.len();
            let to = *index.entry(v).or_insert(next);
            counts.resize(index.len(), 0);
            counts[from] += 1;
            endpoints.push((from, to));
        }

        let mut adjacency = Adjacency {
            offsets: Vec::with_capacity(counts.len() + 1),
            edges: vec![(0, 0, 0); self.data.len()],
            data: self.data.values().collect(),
            #[cfg(test)]
            nodes: vec![0; counts.len()],
        };
        #[cfg(test)]
        for (&node, &idx) in &index {
            adjacency.nodes[idx] = node;
        }
        drop(index);

        adjacency.offsets.push(0);
        for count in &mut counts {
            adjacency
                .offsets
                .push(adjacency.offsets.last().unwrap() + *count);
            *count = 0;
        }

        // Group edges by source without changing the neighbour order within each group.
        for ((from, to), edge) in endpoints.into_iter().zip(self.data.values()) {
            let pos = adjacency.offsets[from] + counts[from];
            let adjusted = (edge.adj_dist * 100.0).round() as u32;
            let intact = (edge.geo_dist * 100.0).round() as u32;
            adjacency.edges[pos] = (to, adjusted, intact);
            adjacency.data[pos] = edge;
            counts[from] += 1;
        }

        adjacency
    }
}

/// Heap entry ordered by cost alone, lowest first, as in `pathfinding::dijkstra_all`,
/// so equal-cost entries pop in the same order and ties resolve the same way.
struct Queued {
    cost: u32,
    index: usize,
}

impl PartialEq for Queued {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for Queued {}

impl PartialOrd for Queued {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Queued {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost)
    }
}

/// Shortest-path state indexed directly by local node index.
pub struct Tree {
    pub parent: Vec<usize>,
    pub edge: Vec<usize>,
    pub cost: Vec<u32>,
    pub order: Vec<usize>, // settlement order
}

/// Dijkstra over integer weights. Keeps the existing cost-only heap ordering and
/// strict improvement rule, so parents and settlement order are unchanged.
/// Adapted from `run_dijkstra` in the pathfinding crate (Samuel Tardieu, MIT OR Apache-2.0).
fn dijkstra(adjacency: &Adjacency<'_>, weight: impl Fn(&(usize, u32, u32)) -> u32) -> Tree {
    let num_nodes = adjacency.offsets.len() - 1;
    let mut tree = Tree {
        parent: vec![usize::MAX; num_nodes],
        edge: vec![0; num_nodes],
        cost: vec![0; num_nodes],
        order: Vec::with_capacity(num_nodes),
    };
    // Index zero is the source. The parent marks whether a node has been reached;
    // using u32::MAX as an unseen cost would discard valid paths with that cost.
    tree.parent[0] = 0;

    let mut heap = BinaryHeap::new();
    heap.push(Queued { cost: 0, index: 0 });
    while let Some(Queued { cost, index }) = heap.pop() {
        // Skip entries superseded by a cheaper path
        if cost > tree.cost[index] {
            continue;
        }
        tree.order.push(index);
        for pos in adjacency.offsets[index]..adjacency.offsets[index + 1] {
            let edge = &adjacency.edges[pos];
            // Checked: a wrapped cost would re-settle nodes and silently double-count them
            let new_cost = cost
                .checked_add(weight(edge))
                .expect("Path cost overflow; check max_cost and resistance scaling!");
            let next = edge.0;
            // Only a strictly cheaper path replaces the parent
            if tree.parent[next] != usize::MAX && tree.cost[next] <= new_cost {
                continue;
            }
            tree.parent[next] = index;
            tree.edge[next] = pos;
            tree.cost[next] = new_cost;
            heap.push(Queued {
                cost: new_cost,
                index: next,
            });
        }
    }

    tree
}

pub trait GraphDijkstraExt {
    fn paths(&self) -> Vec<EdgeData>;
}

impl GraphDijkstraExt for Graph {
    /// Calculate adjusted and intact paths using the same adjacency list.
    fn paths(&self) -> Vec<EdgeData> {
        let adjacency = self.to_adjacency();
        let altered = dijkstra(&adjacency, |&(_, adjusted, _)| adjusted);
        let intact = dijkstra(&adjacency, |&(_, _, intact)| intact);

        let mut distances = vec![0.0f32; altered.cost.len()];
        let mut cell_paths = Vec::with_capacity(altered.order.len() - 1);

        // Settlement order is fixed and parents settle before their children, so sums keep
        // the source-to-target addition order; order[0] is the source itself.
        for &i in &altered.order[1..] {
            let p = altered.parent[i];
            let edge = adjacency.data[altered.edge[i]];
            distances[i] = distances[p] + edge.adj_dist;

            let dist_intact = intact.cost[i] as f32 / 100.0;
            cell_paths.push(EdgeData {
                adj_dist: distances[i],
                geo_dist: dist_intact,
                condition: edge.condition,
                pa: edge.pa,
                num_cells: edge.num_cells,
                similarities: Rc::clone(&edge.similarities),
            });
        }

        cell_paths
    }
}

/// Both search trees and the per-target paths of one graph, for reference comparisons.
pub struct Routed<'a> {
    pub adjacency: Adjacency<'a>,
    pub altered: Tree,
    pub intact: Tree,
    pub paths: Vec<EdgeData>,
}

/// Canonical-reference hooks: the baseline adjacency and search, with the edge order supplied
/// by the caller instead of taken from the edge table.
impl Graph {
    /// Same as `to_adjacency`, but edges are grouped by source in the order of `keys`.
    pub fn to_adjacency_in(&self, keys: &[(NodeId, NodeId)]) -> Adjacency<'_> {
        assert_eq!(keys.len(), self.data.len(), "Key order must cover every edge once.");
        let mut index = FxHashMap::default();
        index.insert(self.source, 0);
        let mut counts = vec![0usize];
        let mut endpoints = Vec::with_capacity(keys.len());

        for &(u, v) in keys {
            let next = index.len();
            let from = *index.entry(u).or_insert(next);
            let next = index.len();
            let to = *index.entry(v).or_insert(next);
            counts.resize(index.len(), 0);
            counts[from] += 1;
            endpoints.push((from, to));
        }

        let mut adjacency = Adjacency {
            offsets: Vec::with_capacity(counts.len() + 1),
            edges: vec![(0, 0, 0); keys.len()],
            data: keys.iter().map(|key| &self.data[key]).collect(),
            nodes: vec![0; counts.len()],
        };
        for (&node, &idx) in &index {
            adjacency.nodes[idx] = node;
        }
        drop(index);

        adjacency.offsets.push(0);
        for count in &mut counts {
            adjacency
                .offsets
                .push(adjacency.offsets.last().unwrap() + *count);
            *count = 0;
        }

        for ((from, to), key) in endpoints.into_iter().zip(keys) {
            let edge = &self.data[key];
            let pos = adjacency.offsets[from] + counts[from];
            let adjusted = (edge.adj_dist * 100.0).round() as u32;
            let intact = (edge.geo_dist * 100.0).round() as u32;
            adjacency.edges[pos] = (to, adjusted, intact);
            adjacency.data[pos] = edge;
            counts[from] += 1;
        }

        adjacency
    }

    /// The body of `paths`, run on a given adjacency and keeping both trees.
    pub fn route<'a>(&'a self, adjacency: Adjacency<'a>) -> Routed<'a> {
        let altered = dijkstra(&adjacency, |&(_, adjusted, _)| adjusted);
        let intact = dijkstra(&adjacency, |&(_, _, intact)| intact);

        let mut distances = vec![0.0f32; altered.cost.len()];
        let mut cell_paths = Vec::with_capacity(altered.order.len() - 1);

        for &i in &altered.order[1..] {
            let p = altered.parent[i];
            let edge = adjacency.data[altered.edge[i]];
            distances[i] = distances[p] + edge.adj_dist;

            let dist_intact = intact.cost[i] as f32 / 100.0;
            cell_paths.push(EdgeData {
                adj_dist: distances[i],
                geo_dist: dist_intact,
                condition: edge.condition,
                pa: edge.pa,
                num_cells: edge.num_cells,
                similarities: Rc::clone(&edge.similarities),
            });
        }

        Routed {
            adjacency,
            altered,
            intact,
            paths: cell_paths,
        }
    }
}

/// Return distance values and the condition/similarity of the last segment
#[cfg(test)]
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
    use super::super::metrics;
    use pathfinding::prelude::{build_path, dijkstra_all};

    fn reference(graph: &Graph) -> Vec<EdgeData> {
        let graph_int = graph.to_adjacency();
        let mut nodes = Vec::with_capacity(2);
        let mut order = Vec::new();
        for adjusted in [true, false] {
            let edge_counts = graph.count_edges();
            let mut adjacency = FxHashMap::default();
            adjacency.reserve(edge_counts.len());
            for (&node, &count) in &edge_counts {
                adjacency.insert(node, Vec::with_capacity(count));
            }
            for (&(u, v), edge) in &graph.data {
                let dist = if adjusted {
                    edge.adj_dist
                } else {
                    edge.geo_dist
                };
                let weight = (dist * 100.0).round() as u32;
                if let Some(neighbors) = adjacency.get_mut(&u) {
                    neighbors.push((v, weight));
                }
            }
            let successors = |node: &NodeId| {
                if adjusted && *node != graph.source {
                    order.push(*node);
                }
                adjacency.get(node).into_iter().flatten().copied()
            };
            let paths = dijkstra_all(&graph.source, successors);

            // Dense search must match the original hash-based reference, including ties
            let weight = |e: &(usize, u32, u32)| if adjusted { e.1 } else { e.2 };
            let tree = dijkstra(&graph_int, weight);
            assert_eq!(tree.order.len(), paths.len() + 1);
            for &i in &tree.order[1..] {
                let parent = graph_int.nodes[tree.parent[i]];
                assert_eq!(paths[&graph_int.nodes[i]], (parent, tree.cost[i]));
            }
            if adjusted {
                let settled: Vec<NodeId> = tree.order[1..]
                    .iter()
                    .map(|&i| graph_int.nodes[i])
                    .collect();
                assert_eq!(settled, order);
            }
            nodes.push(paths);
        }

        order
            .iter()
            .map(|node| {
                let path = build_path(node, &nodes[0]);
                let dist_intact = nodes[1][node].1 as f32 / 100.0;
                path_distance(graph, &path, dist_intact)
            })
            .collect()
    }

    #[test]
    fn matches_reference() {
        for size in [2, 5, 11, 25] {
            for zero_cost in [false, true] {
                let mut graph = Graph::new(Some(size * size * 8));
                for i in 0..size as i32 {
                    for j in 0..size as i32 {
                        let u = (i as usize * size + j as usize) as NodeId;
                        for di in -1..=1 {
                            for dj in -1..=1 {
                                let (ni, nj) = (i + di, j + dj);
                                if (di == 0 && dj == 0)
                                    || ni < 0
                                    || nj < 0
                                    || ni >= size as i32
                                    || nj >= size as i32
                                {
                                    continue;
                                }
                                let v = (ni as usize * size + nj as usize) as NodeId;
                                let base = if zero_cost { 0.001 } else { 1.0 };
                                graph.add_node(
                                    (u, v),
                                    EdgeData {
                                        adj_dist: base + ((u * 17 + v * 11) % 20) as f32 * 0.0001,
                                        geo_dist: if di != 0 && dj != 0 { 1.4142 } else { 1.0 },
                                        condition: (v % 9 + 1) as f32 / 10.0,
                                        pa: (v % 3) as f32 / 2.0,
                                        num_cells: (v % 5 + 1) as f32,
                                        similarities: Rc::new(vec![
                                            Some(1.0),
                                            Some((v % 7) as f32 / 7.0),
                                            None,
                                        ]),
                                    },
                                );
                            }
                        }
                    }
                }

                let expected = reference(&graph);
                for _ in 0..3 {
                    let actual = graph.paths();
                    assert_eq!(actual.len(), expected.len());
                    for (a, b) in actual.iter().zip(&expected) {
                        assert_eq!(
                            [a.adj_dist, a.geo_dist, a.condition, a.pa, a.num_cells]
                                .map(f32::to_bits),
                            [b.adj_dist, b.geo_dist, b.condition, b.pa, b.num_cells]
                                .map(f32::to_bits),
                        );
                        assert_eq!(a.similarities, b.similarities);
                    }
                    for lambda in [2.0, 20.0, 200.0] {
                        assert_eq!(
                            metrics::connectedness(&actual, lambda).to_bits(),
                            metrics::connectedness(&expected, lambda).to_bits()
                        );
                        assert_eq!(
                            metrics::beri_score(&actual, lambda).to_bits(),
                            metrics::beri_score(&expected, lambda).to_bits()
                        );
                    }
                }
            }
        }
    }

    // The reference hooks must reproduce `paths` exactly when given the edge-table order
    #[test]
    fn route_matches_paths() {
        for zero_cost in [false, true] {
            let mut graph = Graph::new(None);
            for u in 0..40u64 {
                for v in [u + 1, u + 3, (u * 7) % 41] {
                    if v == u {
                        continue;
                    }
                    let base = if zero_cost { 0.001 } else { 1.0 };
                    graph.add_node(
                        (u, v),
                        EdgeData {
                            adj_dist: base + ((u * 13 + v) % 7) as f32 * 0.25,
                            geo_dist: 1.0 + (v % 3) as f32,
                            condition: (v % 9 + 1) as f32 / 10.0,
                            pa: 1.0,
                            num_cells: (v % 5 + 1) as f32,
                            similarities: Rc::new(vec![Some(1.0), None]),
                        },
                    );
                }
            }
            let keys: Vec<(NodeId, NodeId)> = graph.data.keys().copied().collect();
            let routed = graph.route(graph.to_adjacency_in(&keys));
            let expected = graph.paths();
            assert_eq!(routed.paths.len(), expected.len());
            for (a, b) in routed.paths.iter().zip(&expected) {
                assert_eq!(
                    [a.adj_dist, a.geo_dist, a.condition, a.num_cells].map(f32::to_bits),
                    [b.adj_dist, b.geo_dist, b.condition, b.num_cells].map(f32::to_bits)
                );
            }
        }
    }

    #[test]
    fn keeps_sum_order() {
        let mut graph = Graph::new(None);
        for (u, v, dist) in [(0, 1, 16_777_216.0), (1, 2, 1.0), (2, 3, 1.0)] {
            graph.add_node(
                (u, v),
                EdgeData {
                    adj_dist: dist,
                    geo_dist: 1.0,
                    condition: v as f32,
                    pa: 1.0,
                    num_cells: 1.0,
                    similarities: Rc::new(Vec::new()),
                },
            );
        }
        let actual = graph.paths();
        let expected = reference(&graph);
        assert_eq!(
            actual.last().unwrap().adj_dist.to_bits(),
            expected.last().unwrap().adj_dist.to_bits()
        );
        assert_eq!(actual.last().unwrap().adj_dist, 16_777_216.0);
    }

    // Two edges of 3e9 scaled cost exceed u32::MAX; must fail in every build profile
    #[test]
    #[should_panic(expected = "Path cost overflow")]
    fn rejects_overflow() {
        let mut graph = Graph::new(None);
        for (u, v) in [(0, 1), (1, 2)] {
            graph.add_node(
                (u, v),
                EdgeData {
                    adj_dist: 3.0e7,
                    geo_dist: 1.0,
                    condition: 1.0,
                    pa: 1.0,
                    num_cells: 1.0,
                    similarities: Rc::new(Vec::new()),
                },
            );
        }
        graph.paths();
    }

    #[test]
    fn empty_and_isolated() {
        let mut graph = Graph::new(None);
        graph.source = (8 << 48) | (1_000_003 << 24) | 2_000_005;
        let source = graph.source;
        assert!(graph.paths().is_empty());
        graph.add_node(
            (source, source + 1),
            EdgeData {
                adj_dist: 1.234,
                geo_dist: 0.987,
                condition: 0.5,
                pa: 1.0,
                num_cells: 4.0,
                similarities: Rc::new(vec![Some(0.75)]),
            },
        );
        graph.add_node(
            (source + 2, source + 3),
            graph.get(&(source, source + 1)).unwrap().clone(),
        );
        let adjacency = graph.to_adjacency();
        assert_eq!(adjacency.nodes.len(), 4);
        assert_eq!(adjacency.offsets.len(), 5);
        let expected = reference(&graph);
        let paths = graph.paths();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].adj_dist.to_bits(), expected[0].adj_dist.to_bits());
        assert_eq!(paths[0].adj_dist.to_bits(), 1.234f32.to_bits());
        assert_eq!(paths[0].geo_dist.to_bits(), 0.99f32.to_bits());
        assert_eq!(paths[0].similarities.as_ref(), &[Some(0.75)]);
    }

    #[test]
    fn reaches_max_cost() {
        let adjacency = Adjacency {
            offsets: vec![0, 1, 1, 1],
            edges: vec![(1, u32::MAX, u32::MAX)],
            data: Vec::new(),
            nodes: vec![0, 1, 2],
        };
        let tree = dijkstra(&adjacency, |e| e.1);
        assert_eq!(tree.order, vec![0, 1]);
        assert_eq!(tree.cost[1], u32::MAX);
        assert_eq!(tree.parent[1], 0);
        assert_eq!(tree.parent[2], usize::MAX);
    }

    #[test]
    fn sums_raw_adj_dist() {
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

        let paths = graph.paths();
        let edge = &paths[1];

        assert!((edge.adj_dist - 8.0).abs() < 1.0e-6);
        assert!((edge.geo_dist - 6.0).abs() < 1.0e-6);
        assert!((edge.condition - 0.25).abs() < 1.0e-6);
        assert!((edge.pa - 0.5).abs() < 1.0e-6);
        assert!((edge.num_cells - 7.0).abs() < 1.0e-6);
        assert_eq!(edge.similarities.as_ref(), &[Some(1.0)]);
    }
}
