use crate::graph::{EdgeData, Graph, NodeId};
use pathfinding::prelude::dijkstra_all;
use rustc_hash::FxHashMap;
use std::rc::Rc;

/// Convert edge list to adjacency list format to be injested by dijksta_all function via successor
/// Converts f32 weights to u32 by multiplying with 100 and rounding to be used in Dijkstra
/// HashMap<u, Vec<(v, adjusted, intact)>>
/// u: source, v: destination
impl Graph {
    #[inline]
    fn to_adjacency(&self) -> FxHashMap<NodeId, Vec<(NodeId, u32, u32)>> {
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
            let adjusted = (edge.adj_dist * 100.0).round() as u32;
            let intact = (edge.geo_dist * 100.0).round() as u32;

            if let Some(neighbors) = adjacency.get_mut(&u) {
                neighbors.push((v, adjusted, intact));
            }
        }

        adjacency
    }
}

pub trait GraphDijkstraExt {
    fn paths(&self) -> Vec<EdgeData>;
}

impl GraphDijkstraExt for Graph {
    /// Calculate adjusted and intact paths using the same adjacency list.
    fn paths(&self) -> Vec<EdgeData> {
        let graph_int = self.to_adjacency();
        // Keep settlement order for repeatable summation.
        let mut order = Vec::with_capacity(graph_int.len());
        let successors = |node: &NodeId| {
            if *node != self.source {
                order.push(*node);
            }
            graph_int
                .get(node)
                .into_iter()
                .flatten()
                .map(|&(v, adjusted, _)| (v, adjusted))
        };
        let nodes_altered = dijkstra_all(&self.source, successors);
        // `order` relies on dijkstra_all expanding each reachable node exactly once.
        assert_eq!(order.len(), nodes_altered.len(), "Settlement order mismatch!");
        let successors = |node: &NodeId| {
            graph_int
                .get(node)
                .into_iter()
                .flatten()
                .map(|&(v, _, intact)| (v, intact))
        };
        let nodes_intact = dijkstra_all(&self.source, successors);
        drop(graph_int);

        let mut distances = FxHashMap::default();
        distances.reserve(nodes_altered.len() + 1);
        distances.insert(self.source, 0.0f32);
        let mut cell_paths = Vec::with_capacity(nodes_altered.len());

        // Parents are settled first; keep the original source-to-target addition order.
        for node in order {
            let parent = nodes_altered[&node].0;
            let edge = self.get(&(parent, node)).expect("Edge not found!");
            let dist_adjusted = distances[&parent] + edge.adj_dist;
            distances.insert(node, dist_adjusted);

            let dist_intact = nodes_intact[&node].1 as f32 / 100.0;
            cell_paths.push(EdgeData {
                adj_dist: dist_adjusted,
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
    use crate::metrics;
    use pathfinding::prelude::build_path;

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
            let successors = |node: &NodeId| {
                graph_int
                    .get(node)
                    .into_iter()
                    .flatten()
                    .map(|&(v, adj, geo)| (v, if adjusted { adj } else { geo }))
            };
            assert_eq!(paths, dijkstra_all(&graph.source, successors));
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

    #[test]
    fn empty_and_isolated() {
        let mut graph = Graph::new(None);
        assert!(graph.paths().is_empty());
        graph.add_node(
            (0, 1),
            EdgeData {
                adj_dist: 1.234,
                geo_dist: 0.987,
                condition: 0.5,
                pa: 1.0,
                num_cells: 4.0,
                similarities: Rc::new(vec![Some(0.75)]),
            },
        );
        graph.add_node((2, 3), graph.get(&(0, 1)).unwrap().clone());
        let paths = graph.paths();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].adj_dist.to_bits(), 1.234f32.to_bits());
        assert_eq!(paths[0].geo_dist.to_bits(), 0.99f32.to_bits());
        assert_eq!(paths[0].similarities.as_ref(), &[Some(0.75)]);
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
