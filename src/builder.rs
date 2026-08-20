use crate::affine::Affine;
use crate::graph::{Graph, NodeId};
use crate::window::{FocalWindow, WindowMode};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// Build a graph strcut from a multi-level window data
impl Graph {
    #[inline]
    pub fn from_data(
        i_base: i32,
        j_base: i32,
        factor: f32,
        windows: &HashMap<i32, FocalWindow>,
        transforms: &HashMap<i32, Affine>,
        geographic: bool,
        offsets: (usize, usize),
        window_mode: WindowMode,
        pa_to_pa: bool,
    ) -> Self {
        // Pre-compute queen-case neighbour indices
        const COLS: [i32; 8] = [0, 1, 0, -1, 1, 1, -1, -1];
        const ROWS: [i32; 8] = [1, 0, -1, 0, 1, -1, 1, -1];

        // Get and sort levels
        let mut levels: Vec<i32> = windows.keys().cloned().collect();
        levels.sort_unstable(); // faster than sort()

        let max_level = *levels.last().unwrap_or(&1);

        // Pre-allocate with capacity for better performance
        let guess_size = windows
            .values()
            .map(|win| win.i_array.len() * 8)
            .sum::<usize>();
        let mut graph_temp = Graph::new(Some(guess_size));

        // Node mappings of the next level; get level 2 as it's the first higher level
        let size_i = windows.get(&2).map(|win| win.i_array.len()).unwrap_or(36);
        let mut node_mapping_higher = HashMap::with_capacity(size_i);

        // Edge indices as HashSet for faster lookups
        let mut all_edge_indices: HashMap<i32, HashSet<(i32, i32)>> = HashMap::new();
        // Pre-compute edge indices for all levels
        for (&level, win) in windows {
            let edge_indices = if window_mode == WindowMode::Circular {
                get_circular_edge_indices(win)
            } else {
                get_edge_indices(&win.i_array, &win.j_array)
            };
            all_edge_indices.insert(level, edge_indices);
        }

        // Process each level
        'outer: for (iter_level, &level) in levels.iter().enumerate() {
            // Only proceed if level is there..
            if let Some(levelwin) = windows.get(&level) {
                let num_cell = levelwin.i_array.len();
                let edge_indices = all_edge_indices
                    .get(&level)
                    .expect("Level not found in edge indices.");

                let level_affine: &Affine = transforms
                    .get(&level)
                    .expect("Missing level in Affine set.");

                // Generate or update node mappings
                let node_mapping = if iter_level == 0 {
                    // First level node mapping
                    let nm = create_node_mapping(levelwin, level, offsets);
                    // Find the base node index only once
                    if let Some((_, (u, _, _, _, _, _))) = nm.get_key_value(&(i_base, j_base)) {
                        graph_temp.source = *u;
                    }

                    nm
                } else {
                    // Reuse mapping already calculated for the higher level in previous round
                    std::mem::take(&mut node_mapping_higher)
                };

                // Pre-compute higher level node mapping if needed
                if level < max_level {
                    let higher_level = level * 2;
                    if let Some(higherwin) = windows.get(&higher_level) {
                        node_mapping_higher = create_node_mapping(higherwin, higher_level, offsets);
                    }
                }

                // Process all cells in a level and the higher neighbours of the edge
                for cell_idx in 0..num_cell {
                    let i = levelwin.i_array[cell_idx];
                    let j = levelwin.j_array[cell_idx];
                    let u = node_mapping
                        .get(&(i, j))
                        .expect("Node not found in node mapping.")
                        .0;

                    // Process neighbors at the current level
                    // Modify the graph, and return true if cell is isolated
                    let was_isolated = graph_temp.neighbours(
                        i,
                        j,
                        u,
                        &COLS,
                        &ROWS,
                        factor,
                        &node_mapping,
                        &level_affine,
                        geographic,
                        pa_to_pa,
                    );

                    // For source nodes with no neighbours (isolated pixel, e.g tiny islands), add duplicated values
                    // and break the outer loop to avoid processing the rest of levels for the current graph
                    if was_isolated {
                        break 'outer;
                    }

                    // Process connections to the next level (e.g. from level 2 to level 4)
                    if level < max_level && edge_indices.contains(&(i, j)) {
                        graph_temp.fringe(
                            i,
                            j,
                            factor,
                            level,
                            &node_mapping,
                            &node_mapping_higher,
                            &transforms,
                            geographic,
                            offsets,
                        );
                    }
                }
            }
        }

        graph_temp
    }
}

/// Create node mapping (the unique ID of each node/pixel)
/// This is done per level/resolution in a window;
/// (i, j) (id, cond, trav, count, pa, sims)
fn create_node_mapping(
    win: &FocalWindow,
    level: i32,
    offsets: (usize, usize),
) -> HashMap<(i32, i32), (NodeId, f32, f32, f32, f32, Rc<Vec<Option<f32>>>)> {
    let num_sims = win.sims.len();
    let (tile_row0_u, tile_col0_u) = offsets;
    let level_origin_i = (tile_row0_u as i32).div_euclid(level);
    let level_origin_j = (tile_col0_u as i32).div_euclid(level);

    let mut mapping = HashMap::with_capacity(win.i_array.len());

    for (idx, (&i_val, &j_val)) in win.i_array.iter().zip(win.j_array.iter()).enumerate() {
        // Collect this cell's similarity across all scenarios
        let mut sim_vals = Vec::with_capacity(num_sims);
        for sim_vec in &win.sims {
            sim_vals.push(sim_vec[idx]);
        }
        let sim_vals = Rc::new(sim_vals);
        let global_i = level_origin_i + i_val;
        let global_j = level_origin_j + j_val;
        let node_id = make_node_id(level, global_i, global_j);

        // Traversal value drives the path weight `w`: the resistance-derived value when a
        // resistance raster was supplied, otherwise the condition value (so the weight is
        // bit-identical to a resistance-free run).
        let trav = match &win.trav_values {
            Some(tv) => tv[idx],
            None => win.values[idx],
        };

        mapping.insert(
            (i_val, j_val),
            (node_id, win.values[idx], trav, win.counts[idx], win.pa_values[idx], sim_vals),
        );
    }

    mapping
}

/// Create a tile-invariant node id from (level, global_row, global_col).
/// Layout in 64 bits: [level:16 | row:24 | col:24]
#[inline]
fn make_node_id(level: i32, global_i: i32, global_j: i32) -> NodeId {
    let l = level as i64;
    let r = global_i as i64;
    let c = global_j as i64;

    if l < 0 || r < 0 || c < 0 {
        panic!(
            "Negative node coordinates are not supported: level={}, row={}, col={}",
            level, global_i, global_j
        );
    }

    let l = l as u64;
    let r = r as u64;
    let c = c as u64;

    if l > 0xFFFF || r > 0xFF_FFFF || c > 0xFF_FFFF {
        panic!(
            "Node id overflow for (level,row,col)=({},{},{}); exceeds 16/24/24-bit layout",
            level, global_i, global_j
        );
    }

    (l << 48) | (r << 24) | c
}

/// Efficiently get edge cells in a level neighbourhood using HashSet
fn get_edge_indices(i_arr: &[i32], j_arr: &[i32]) -> HashSet<(i32, i32)> {
    let i_min = *i_arr.iter().min().unwrap_or(&0);
    let i_max = *i_arr.iter().max().unwrap_or(&0);
    let j_min = *j_arr.iter().min().unwrap_or(&0);
    let j_max = *j_arr.iter().max().unwrap_or(&0);

    // Pre-allocate approximately the right size
    let perimeter = 2 * (i_max - i_min + j_max - j_min);
    let mut edge_set = HashSet::with_capacity(perimeter as usize);

    // Use iterator for better cache locality
    i_arr
        .iter()
        .zip(j_arr.iter())
        .filter(|(&i, &j)| i == i_min || i == i_max || j == j_min || j == j_max)
        .for_each(|(&i, &j)| {
            edge_set.insert((i, j));
        });

    edge_set
}

/// Circular windows need the full curved outer perimeter, not just min/max
/// rows/columns. The window builder computes these cells geometrically from
/// rectangle-circle overlap so cross-level links are placed around the ring.
fn get_circular_edge_indices(win: &FocalWindow) -> HashSet<(i32, i32)> {
    win.edge_cells
        .as_ref()
        .map(|edges| edges.iter().copied().collect())
        .unwrap_or_default()
}
