use crate::graph::{Csr, Edge, NONE};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Heap entry ordered by cost alone, lowest first, as in `pathfinding::dijkstra_all`,
/// so equal-cost entries pop in the same order and ties resolve the same way.
struct Queued {
    cost: u32,
    index: u32,
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

/// Shortest-path state indexed by node; `parent` is NONE until a node is reached.
#[derive(Default)]
pub struct Tree {
    pub parent: Vec<u32>,
    pub edge: Vec<u32>, // CSR position of the edge into each node
    pub cost: Vec<u32>,
    pub order: Vec<u32>, // settlement order
}

impl Tree {
    fn reserve(&mut self, nodes: usize) {
        self.parent.reserve(nodes);
        self.edge.reserve(nodes);
        self.cost.reserve(nodes);
        self.order.reserve(nodes);
    }

    fn bytes(&self) -> usize {
        4 * (self.parent.capacity() + self.edge.capacity() + self.cost.capacity() + self.order.capacity())
    }
}

/// Dijkstra over integer weights with the cost-only heap ordering and strict improvement
/// rule of the legacy search, so parents and settlement order follow the CSR edge order.
/// Adapted from `run_dijkstra` in the pathfinding crate (Samuel Tardieu, MIT OR Apache-2.0).
fn dijkstra(
    csr: &Csr,
    source: u32,
    tree: &mut Tree,
    heap: &mut BinaryHeap<Queued>,
    weight: impl Fn(&Edge) -> u32,
) {
    let num_nodes = csr.offsets.len() - 1;
    tree.parent.clear();
    tree.parent.resize(num_nodes, NONE);
    tree.edge.clear();
    tree.edge.resize(num_nodes, 0);
    tree.cost.clear();
    tree.cost.resize(num_nodes, 0);
    tree.order.clear();
    // The parent marks whether a node has been reached; using u32::MAX as an unseen cost
    // would discard valid paths with that cost.
    tree.parent[source as usize] = source;

    heap.clear();
    heap.push(Queued { cost: 0, index: source });
    while let Some(Queued { cost, index }) = heap.pop() {
        // Skip entries superseded by a cheaper path
        if cost > tree.cost[index as usize] {
            continue;
        }
        tree.order.push(index);
        let (start, end) = (csr.offsets[index as usize], csr.offsets[index as usize + 1]);
        for pos in start..end {
            let edge = &csr.edges[pos as usize];
            // Checked: a wrapped cost would re-settle nodes and silently double-count them
            let new_cost = cost
                .checked_add(weight(edge))
                .expect("Path cost overflow; check max_cost and resistance scaling!");
            let next = edge.to as usize;
            // Only a strictly cheaper path replaces the parent
            if tree.parent[next] != NONE && tree.cost[next] <= new_cost {
                continue;
            }
            tree.parent[next] = index;
            tree.edge[next] = pos;
            tree.cost[next] = new_cost;
            heap.push(Queued {
                cost: new_cost,
                index: edge.to,
            });
        }
    }
}

/// Destinations in adjusted settlement order (source excluded) with the raw adjusted
/// distance summed along the adjusted tree and the intact path cost in kilometres.
#[derive(Default)]
pub struct Paths {
    pub node: Vec<u32>,
    pub adj: Vec<f32>,
    pub geo: Vec<f32>,
}

impl Paths {
    pub fn len(&self) -> usize {
        self.node.len()
    }

    pub fn is_empty(&self) -> bool {
        self.node.is_empty()
    }

    pub fn clear(&mut self) {
        self.node.clear();
        self.adj.clear();
        self.geo.clear();
    }
}

/// Reusable routing state: both trees, the heap, per-node path sums and the path stream.
#[derive(Default)]
pub struct Router {
    pub altered: Tree,
    pub intact: Tree,
    heap: BinaryHeap<Queued>,
    sums: Vec<f32>,
    pub paths: Paths,
}

impl Router {
    /// Adjusted and intact searches over the same CSR from `source`.
    pub fn route(&mut self, csr: &Csr, source: u32) {
        dijkstra(csr, source, &mut self.altered, &mut self.heap, |e| e.adj_cost);
        dijkstra(csr, source, &mut self.intact, &mut self.heap, |e| e.geo_cost);

        self.sums.clear();
        self.sums.resize(csr.offsets.len() - 1, 0.0);
        self.paths.clear();

        // Settlement order is fixed and parents settle before their children, so sums keep
        // the source-to-target addition order; order[0] is the source itself.
        for &i in &self.altered.order[1..] {
            let (i, p) = (i as usize, self.altered.parent[i as usize] as usize);
            let edge = &csr.edges[self.altered.edge[i] as usize];
            self.sums[i] = self.sums[p] + edge.adj;

            self.paths.node.push(i as u32);
            self.paths.adj.push(self.sums[i]);
            self.paths.geo.push(self.intact.cost[i] as f32 / 100.0);
        }
    }

    /// Capacity for graphs of up to `nodes` nodes and `edges` edges
    pub fn reserve(&mut self, nodes: usize, edges: usize) {
        self.altered.reserve(nodes);
        self.intact.reserve(nodes);
        self.heap.reserve(edges + 1);
        self.sums.reserve(nodes);
        self.paths.node.reserve(nodes);
        self.paths.adj.reserve(nodes);
        self.paths.geo.reserve(nodes);
    }

    /// No destinations: an empty graph or a source outside its window
    pub fn clear(&mut self) {
        self.altered.order.clear();
        self.intact.order.clear();
        self.paths.clear();
    }

    pub fn bytes(&self) -> usize {
        self.altered.bytes()
            + self.intact.bytes()
            + self.heap.capacity() * std::mem::size_of::<Queued>()
            + self.sums.capacity() * 4
            + 4 * (self.paths.node.capacity() + self.paths.adj.capacity() + self.paths.geo.capacity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::Rng;
    use pathfinding::prelude::dijkstra_all;
    use std::cell::RefCell;

    fn csr(lists: &[Vec<(u32, f32)>]) -> Csr {
        let mut csr = Csr::default();
        csr.clear();
        for list in lists {
            for &(to, dist) in list {
                csr.edges.push(Edge::new(to, 1.0, dist));
            }
            csr.close();
        }
        csr
    }

    /// Queen-linked grid with a shuffled per-node edge order, many ties and zero costs.
    fn random_graph(rng: &mut Rng, size: usize, zero_cost: bool) -> Csr {
        let mut lists = vec![Vec::new(); size * size];
        for i in 0..size as i32 {
            for j in 0..size as i32 {
                let u = i as usize * size + j as usize;
                for di in -1..=1 {
                    for dj in -1..=1 {
                        let (ni, nj) = (i + di, j + dj);
                        let outside = ni < 0 || nj < 0 || ni >= size as i32 || nj >= size as i32;
                        if (di, dj) == (0, 0) || outside {
                            continue;
                        }
                        let v = ni as usize * size + nj as usize;
                        let base = if zero_cost { rng.pick(&[0.0, 0.004, 0.01]) } else { 1.0 };
                        lists[u].push((v as u32, base + rng.below(3) as f32 * 0.01));
                    }
                }
                let n = lists[u].len();
                for k in (1..n).rev() {
                    lists[u].swap(k, rng.below(k + 1));
                }
            }
        }
        csr(&lists)
    }

    // Parents, costs and settlement order match pathfinding's search with successors in CSR
    // order; the legacy edge-table order stays covered by legacy::routing::tests.
    #[test]
    fn matches_pathfinding() {
        let mut rng = Rng::new(7);
        let mut router = Router::default();
        for size in [2, 5, 11, 25] {
            for zero_cost in [false, true] {
                let graph = random_graph(&mut rng, size, zero_cost);
                let source = rng.below(size * size) as u32;
                router.route(&graph, source);
                for adjusted in [true, false] {
                    let order = RefCell::new(Vec::new());
                    let successors = |&u: &u32| {
                        if u != source {
                            order.borrow_mut().push(u);
                        }
                        let (s, e) = (graph.offsets[u as usize], graph.offsets[u as usize + 1]);
                        graph.edges[s as usize..e as usize]
                            .iter()
                            .map(move |e| (e.to, if adjusted { e.adj_cost } else { e.geo_cost }))
                    };
                    let expected = dijkstra_all(&source, successors);
                    let tree = if adjusted { &router.altered } else { &router.intact };
                    assert_eq!(tree.order.len(), expected.len() + 1);
                    for &i in &tree.order[1..] {
                        assert_eq!(expected[&i], (tree.parent[i as usize], tree.cost[i as usize]));
                    }
                    if adjusted {
                        assert_eq!(&tree.order[1..], order.borrow().as_slice());
                    }
                }
            }
        }
    }

    #[test]
    fn keeps_sum_order() {
        let graph = csr(&[vec![(1, 16_777_216.0)], vec![(2, 1.0)], vec![(3, 1.0)], vec![]]);
        let mut router = Router::default();
        router.route(&graph, 0);
        assert_eq!(*router.paths.adj.last().unwrap(), 16_777_216.0);
    }

    // Two edges of 3e9 scaled cost exceed u32::MAX; must fail in every build profile
    #[test]
    #[should_panic(expected = "Path cost overflow")]
    fn rejects_overflow() {
        let graph = csr(&[vec![(1, 3.0e7)], vec![(2, 3.0e7)], vec![]]);
        Router::default().route(&graph, 0);
    }

    #[test]
    fn empty_and_isolated() {
        let mut router = Router::default();
        router.route(&csr(&[vec![]]), 0);
        assert!(router.paths.is_empty());

        // A source whose only edge leads to its fake copy, plus an unreachable edge
        let graph = csr(&[vec![], vec![(2, 0.987)], vec![], vec![(4, 1.0)], vec![]]);
        router.route(&graph, 1);
        assert_eq!(router.paths.node, vec![2]);
        assert_eq!(router.paths.adj[0].to_bits(), 0.987f32.to_bits());
        assert_eq!(router.paths.geo[0].to_bits(), 0.99f32.to_bits());
        assert_eq!(router.altered.parent[3], NONE);
    }

    #[test]
    fn reaches_max_cost() {
        let mut graph = csr(&[vec![(1, 0.0)], vec![], vec![]]);
        graph.edges[0].adj_cost = u32::MAX;
        let mut router = Router::default();
        router.route(&graph, 0);
        assert_eq!(router.altered.order, vec![0, 1]);
        assert_eq!(router.altered.cost[1], u32::MAX);
        assert_eq!(router.altered.parent[1], 0);
        assert_eq!(router.altered.parent[2], NONE);
    }

    #[test]
    fn sums_raw_adj_dist() {
        let mut graph = csr(&[vec![(1, 2.0)], vec![(2, 4.0)], vec![]]);
        graph.edges[0].adj = 3.0;
        graph.edges[1].adj = 5.0;
        let mut router = Router::default();
        router.route(&graph, 0);
        assert_eq!(router.paths.node, vec![1, 2]);
        assert!((router.paths.adj[1] - 8.0).abs() < 1.0e-6);
        assert!((router.paths.geo[1] - 6.0).abs() < 1.0e-6);
    }
}
