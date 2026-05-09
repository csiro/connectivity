use crate::affine::Affine;
use crate::distances;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::rc::Rc;

pub type NodeId = u64;

// The data of Graph edge
#[derive(Debug, Clone)]
pub struct EdgeData {
    pub adj_dist: f32,                      // Condistion-adjusted distance
    pub geo_dist: f32,                      // Geo-distance; no condition adjustment
    pub condition: f32,                     // Raw condition
    pub num_cells: f32,                     // Original Cell count/weight for habitat area
    pub similarities: Rc<Vec<Option<f32>>>, // Similarity of the cell; all scenarios
}

/// The multi-resolution graph type
/// struct { HashMap<(source, destination), EdgeData>, source-node }
#[derive(Debug, Clone)]
pub struct Graph {
    pub data: FxHashMap<(NodeId, NodeId), EdgeData>,
    pub source: NodeId, // should be separate as data-oriented design principal, but cleaner now;
}

impl Graph {
    /// Default constructor (no preallocated capacity)
    pub fn new(cap: Option<usize>) -> Self {
        let mut data = FxHashMap::default();
        if let Some(c) = cap {
            data.reserve(c);
        }
        // An initial source
        let source = 0;

        Self { data, source }
    }

    /// Insert a new node entry
    pub fn add_node(&mut self, key: (NodeId, NodeId), value: EdgeData) {
        self.data.insert(key, value);
    }

    /// Get an entry by reference
    pub fn get(&self, key: &(NodeId, NodeId)) -> Option<&EdgeData> {
        self.data.get(&key)
    }

    /// Count how many outgoing edges each `u` node has.
    pub fn count_edges(&self) -> FxHashMap<NodeId, usize> {
        // Count the source in hashmap to keep the keys unique
        let mut edge_counts = FxHashMap::default();
        edge_counts.reserve(self.data.len());
        for &(u, _) in self.data.keys() {
            *edge_counts.entry(u).or_insert(0) += 1;
        }

        edge_counts
    }

    /// Clear the graph; e.g. in case of isolcated pixels
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

/// Add edges to neighboring cells; if a source is isolated, adds a synthetic edge.
impl Graph {
    #[inline]
    pub fn neighbours(
        &mut self,
        i: i32,
        j: i32,
        u: NodeId, // target node id
        i_ngb: &[i32],
        j_ngb: &[i32],
        factor: f32,
        node_mapping: &HashMap<(i32, i32), (NodeId, f32, f32, Rc<Vec<Option<f32>>>)>,
        transform: &Affine,
        is_wgs: bool,
    ) -> bool {
        // Start with true if it's the source node
        let mut is_isolated = u == self.source;

        // Coordinates of the current node only need to be computed once
        let (x1, y1) = transform.xy(i, j);

        // Iterate over queen-case neighbor offsets (8 adjacent neighbours)
        for (&di, &dj) in i_ngb.iter().zip(j_ngb.iter()) {
            let ni = i + di;
            let nj = j + dj;

            // Check if neighbor exists in node_mapping
            if let Some(&(v, z, c, ref s)) = node_mapping.get(&(ni, nj)) {
                // Distance to adjacent node/cell in kilometers
                let (x2, y2) = transform.xy(ni, nj);
                let dist: f32 = distances::distance_km(x1, y1, x2, y2, is_wgs);
                // Calcualte the weight using max_cost
                let w: f32 = (1.0 - factor) * z + factor;

                // Store weighted distance + similarities
                self.add_node(
                    (u, v),
                    EdgeData {
                        adj_dist: w * dist,
                        geo_dist: dist,
                        condition: z,
                        num_cells: c,
                        similarities: s.clone(),
                    },
                );

                // Not isolated; at least one neighbor found
                is_isolated = false;
            }
        }

        // If the source node is isolated, create a "synthetic" edge so it is connected somewhere
        if is_isolated {
            if let Some(&(_, z, c, ref s)) = node_mapping.get(&(i, j)) {
                // Distance to an adjacent cell in kilometers
                let (x2, y2) = transform.xy(i + 1, j);
                let dist: f32 = distances::distance_km(x1, y1, x2, y2, is_wgs);
                // Calcualte the weight using max_cost
                let w: f32 = (1.0 - factor) * z + factor;
                // Make a fake node
                let fake_v = u + 1;

                // Clear the graph to ensure no isolated edge remains
                self.clear();
                self.add_node(
                    (u, fake_v),
                    EdgeData {
                        adj_dist: w * dist,
                        geo_dist: dist,
                        condition: z,
                        num_cells: c,
                        similarities: s.clone(),
                    },
                );
            }
        }

        is_isolated
    }
}

/// Process connections to the next level (e.g. from level 2 to level 4)
impl Graph {
    #[inline]
    pub fn fringe(
        &mut self,
        i: i32,
        j: i32,
        factor: f32,
        level: i32,
        node_mapping: &HashMap<(i32, i32), (NodeId, f32, f32, Rc<Vec<Option<f32>>>)>,
        node_mapping_higher: &HashMap<(i32, i32), (NodeId, f32, f32, Rc<Vec<Option<f32>>>)>,
        transforms: &HashMap<i32, Affine>,
        is_wgs: bool,
        offsets: (usize, usize),
        include_containing_higher: bool,
    ) {
        if let Some(&(uu, _, _, _)) = node_mapping.get(&(i, j)) {
            let higher_level: i32 = level * 2;

            // Get all higher neighbours at once
            let (higher_neighbours, n_higher_neighbours) =
                get_edge_neighbours(i, j, level, offsets, include_containing_higher);

            // Get the Affines for distance calc
            let transform: &Affine = transforms.get(&level).unwrap();
            let transform_upper: &Affine = transforms.get(&higher_level).unwrap();
            // Get the actual coordinates values for distance calc
            let (x1, y1) = transform.xy(i, j);

            // Use 'ref' to borrow Vec<f32> rather than moving it
            for &(ni, nj) in higher_neighbours[..n_higher_neighbours].iter() {
                // Only if the neghbours are in the higher mapping proceess
                if let Some(&(v, z, c, ref s)) = node_mapping_higher.get(&(ni, nj)) {
                    // Get the actual coordinates of the higher level
                    let (x2, y2) = transform_upper.xy(ni, nj);
                    // Distance in kilometer
                    let dist: f32 = distances::distance_km(x1, y1, x2, y2, is_wgs);

                    // Calcualte the weight using max_cost
                    let w = (1.0 - factor) * z + factor;

                    self.add_node(
                        (uu, v),
                        EdgeData {
                            adj_dist: w * dist,
                            geo_dist: dist,
                            condition: z,
                            num_cells: c,
                            similarities: s.clone(),
                        },
                    );
                }
            }
        }
    }
}

/// Get possible higher-level cells for a current-level edge cell.
///
/// Block windows use only the higher-level cells adjacent to the cell containing
/// the current edge cell. Fractional annulus windows can also overlap that
/// containing higher-level cell at the rim, so annulus modes include it too.
fn get_edge_neighbours(
    i: i32,
    j: i32,
    level: i32,
    offsets: (usize, usize),
    include_containing_higher: bool,
) -> ([(i32, i32); 4], usize) {
    let (tile_row0_u, tile_col0_u) = offsets;
    let tile_row0 = tile_row0_u as i32;
    let tile_col0 = tile_col0_u as i32;

    #[inline]
    fn to_higher_local(idx: i32, level: i32, tile0: i32) -> i32 {
        let curr_origin = tile0.div_euclid(level);
        let higher_origin = tile0.div_euclid(level * 2);
        (curr_origin + idx).div_euclid(2) - higher_origin
    }

    // Higher level cell containing the target cell
    let target_higher = (
        to_higher_local(i, level, tile_row0),
        to_higher_local(j, level, tile_col0),
    );
    // 8 neighbor offsets: N, S, W, E, NW, NE, SW, SE
    const OFFSETS: [(i32, i32); 8] = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ];

    let mut higher_cells = [(0, 0); 4];
    let mut count = 0;

    if include_containing_higher {
        higher_cells[count] = target_higher;
        count += 1;
    }

    for (di, dj) in OFFSETS {
        let ni = i + di;
        let nj = j + dj;
        let higher = (
            to_higher_local(ni, level, tile_row0),
            to_higher_local(nj, level, tile_col0),
        );

        // Skip if it's the same as target's higher cell
        if higher == target_higher {
            continue;
        }

        // Check if we already have this higher cell
        let mut found = false;
        for k in 0..count {
            if higher_cells[k] == higher {
                found = true;
                break;
            }
        }

        // Add if not found and we have space
        if !found && count < higher_cells.len() {
            higher_cells[count] = higher;
            count += 1;
        }
    }

    (higher_cells, count)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_neighbours_preserve_block_exclusion_of_containing_cell() {
        let (cells, count) = get_edge_neighbours(1, 1, 1, (0, 0), false);
        let cells = &cells[..count];

        assert_eq!(count, 3);
        assert!(!cells.contains(&(0, 0)));
        assert!(cells.contains(&(1, 0)));
        assert!(cells.contains(&(0, 1)));
        assert!(cells.contains(&(1, 1)));
    }

    #[test]
    fn edge_neighbours_include_containing_cell_for_annulus_modes() {
        let (cells, count) = get_edge_neighbours(1, 1, 1, (0, 0), true);
        let cells = &cells[..count];

        assert_eq!(count, 4);
        assert!(cells.contains(&(0, 0)));
        assert!(cells.contains(&(1, 0)));
        assert!(cells.contains(&(0, 1)));
        assert!(cells.contains(&(1, 1)));
    }
}
