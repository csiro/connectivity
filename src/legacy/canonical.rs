//! Canonical reference: legacy construction, arithmetic, search and metrics, with each
//! source's out-edges reordered to the documented production order. The order is enumerated
//! from decoded node coordinates, independently of the production builder:
//! 1. same-level queen neighbours in the legacy `neighbours` order, which visits
//!    (di, dj) = (COLS[k], ROWS[k]) for COLS = [0, 1, 0, -1, 1, 1, -1, -1] and
//!    ROWS = [1, 0, -1, 0, 1, -1, 1, -1];
//! 2. links to the next level in `get_edge_neighbours` order.
//! The isolated-source synthetic edge (u, u + 1) is the only edge of its graph, and it is
//! enumerated as the (0, 1) neighbour.
use super::graph::{get_edge_neighbours, Graph, NodeId};
use super::window::FocalWindow;
use super::{map_cells, Prepared};
use ndarray::Array2;
use rustc_hash::FxHashMap;

const COLS: [i32; 8] = [0, 1, 0, -1, 1, 1, -1, -1];
const ROWS: [i32; 8] = [1, 0, -1, 0, 1, -1, 1, -1];

/// Distance override applied once the canonical edge set is fixed: the distance of edge
/// (u, v), flagged when it is the synthetic edge, or `None` to keep the legacy distance.
pub type Inject<'a> = &'a (dyn Fn(NodeId, NodeId, bool) -> Option<f32> + Sync);

/// Decode a legacy node id into (level, global row, global column).
pub fn decode(id: NodeId) -> (i32, i32, i32) {
    ((id >> 48) as i32, ((id >> 24) & 0xFF_FFFF) as i32, (id & 0xFF_FFFF) as i32)
}

/// Encode (level, global row, global column) with the legacy [level:16 | row:24 | col:24] layout.
pub fn encode(level: i32, row: i32, col: i32) -> Option<NodeId> {
    (row >= 0 && col >= 0).then(|| ((level as u64) << 48) | ((row as u64) << 24) | col as u64)
}

/// Every edge of `graph`, grouped by source in ascending node id, in the documented order.
pub fn edge_order(graph: &Graph, offsets: (usize, usize)) -> Vec<(NodeId, NodeId)> {
    let mut sources: Vec<NodeId> = graph.data.keys().map(|&(u, _)| u).collect();
    sources.sort_unstable();
    sources.dedup();

    let origin = |tile0: usize, level: i32| (tile0 as i32).div_euclid(level);
    let mut keys = Vec::with_capacity(graph.data.len());
    for u in sources {
        let (level, row, col) = decode(u);
        for k in 0..8 {
            if let Some(v) = encode(level, row + COLS[k], col + ROWS[k]) {
                if graph.data.contains_key(&(u, v)) {
                    keys.push((u, v));
                }
            }
        }

        let local = (row - origin(offsets.0, level), col - origin(offsets.1, level));
        let (higher, count) = get_edge_neighbours(local.0, local.1, level, offsets);
        for &(hi, hj) in &higher[..count] {
            let up = level * 2;
            let global = (hi + origin(offsets.0, up), hj + origin(offsets.1, up));
            if let Some(v) = encode(up, global.0, global.1) {
                if graph.data.contains_key(&(u, v)) {
                    keys.push((u, v));
                }
            }
        }
    }
    assert_eq!(keys.len(), graph.data.len(), "Canonical order must enumerate every edge.");
    keys
}

/// Traversal value of every window node, keyed by legacy node id.
pub fn traversal(windows: &[(i32, FocalWindow)], offsets: (usize, usize)) -> FxHashMap<NodeId, f32> {
    let mut trav = FxHashMap::default();
    for (level, win) in windows {
        let origin = (
            (offsets.0 as i32).div_euclid(*level),
            (offsets.1 as i32).div_euclid(*level),
        );
        for (idx, (&i, &j)) in win.i_array.iter().zip(&win.j_array).enumerate() {
            let t = match &win.trav_values {
                Some(tv) => tv[idx],
                None => win.values[idx],
            };
            trav.insert(encode(*level, origin.0 + i, origin.1 + j).unwrap(), t);
        }
    }
    trav
}

/// Replace edge distances after the edge set is fixed, recomputing `adj_dist` from the
/// destination's traversal value as the legacy builder does. The synthetic edge's fake node
/// carries the source's values.
pub fn reweight(
    graph: &mut Graph,
    windows: &[(i32, FocalWindow)],
    offsets: (usize, usize),
    factor: f32,
    dist: Inject,
) {
    let trav = traversal(windows, offsets);
    let source_t = trav.get(&graph.source).copied();
    for (&(u, v), edge) in graph.data.iter_mut() {
        if let Some(d) = dist(u, v, !trav.contains_key(&v)) {
            let t = trav.get(&v).copied().or(source_t).expect("Unknown edge target.");
            let w: f32 = (1.0 - factor) * t + factor;
            edge.adj_dist = w * d;
            edge.geo_dist = d;
        }
    }
}

/// Canonical value of cell ij, optionally with injected distances.
pub fn cell(prepared: &Prepared, i: usize, j: usize, inject: Option<Inject>) -> f32 {
    let windows = prepared.windows(i, j);
    let mut graph = prepared.graph(i, j, &windows);
    let keys = edge_order(&graph, prepared.offsets);
    if let Some(dist) = inject {
        reweight(&mut graph, &windows, prepared.offsets, prepared.max_cost, dist);
    }
    let routed = graph.route(graph.to_adjacency_in(&keys));
    prepared.score(&routed.paths)
}

/// Canonical reference output for every unmasked cell.
pub fn conn(
    prepared: &Prepared,
    cond: &Array2<f32>,
    mask: &Array2<bool>,
    inject: Option<Inject>,
) -> Array2<f32> {
    map_cells(cond, mask, |i, j| cell(prepared, i, j, inject))
}
