use std::collections::HashMap;
use std::sync::Arc;
use crate::affine::Affine;
use crate::distances;

/// The multi-resolution graph type
#[derive(Debug, Clone)]
pub struct Graph {
    pub data: HashMap<(u32, u32), (f32, f32, f32, Arc<Vec<f32>>)>,
    pub source: u32, // should be separate as data-oriented design principal, but cleaner now;
}

impl Graph {
    /// Default constructor (no preallocated capacity)
    pub fn new(cap: Option<usize>) -> Self {
        let data = match cap {
            Some(c) => HashMap::with_capacity(c),
            None => HashMap::new(),
        };
        // An initial source
        let source = 0;

        Self { data, source }
    }

    /// Insert a new node entry
    pub fn add_node(
        &mut self,
        key: (u32, u32),
        value: (f32, f32, f32, Arc<Vec<f32>>)
    ) {
        self.data.insert(key, value);
    }

    /// Get an entry by reference
    pub fn get(
        &self,
        key: &(u32, u32),
    ) -> Option<&(f32, f32, f32, Arc<Vec<f32>>)> {
        self.data.get(&key)
    }

    /// Count how many outgoing edges each `u` node has.
    pub fn count_edges(&self) -> HashMap<u32, usize> {
        // Count the source in hashmap to keep the keys unique
        let mut edge_counts: HashMap<u32, usize> = HashMap::new();
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
/// output graph: (u, v) (adj_cond, cond, dist, similarities)
/// u: source, v: destination
impl Graph {
    #[inline]
    pub fn neighbours(
        &mut self,
        i: i32,
        j: i32,
        u: u32, // target node id
        i_ngb: &[i32],
        j_ngb: &[i32],
        factor: f32,
        node_mapping: &HashMap<(i32, i32), (u32, f32, Arc<Vec<f32>>)>,
        transform: &Affine,
        is_wgs: bool,
    ) -> bool {
        // Start with true if it's the source node
        let mut is_isolated = u == self.source;
    
        // Coordinates of the current node only need to be computed once
        let (x1, y1) = transform.xy(j, i);
    
        // Iterate over queen-case neighbor offsets (8 adjacent neighbours)
        for (&di, &dj) in i_ngb.iter().zip(j_ngb.iter()) {
            let ni = i + di;
            let nj = j + dj;
    
            // Check if neighbor exists in node_mapping
            if let Some(&(v, z, ref s)) = node_mapping.get(&(ni, nj)) {
                // Distance to adjacent node/cell in kilometers
                let (x2, y2) = transform.xy(nj, ni);
                let dist: f32 = distances::distance_km(x1, y1, x2, y2, is_wgs);
                // Calcualte the weight using max_cost
                let w: f32 = (1.0 - factor) * z + factor;
    
                // Store weighted distance + similarities
                self.add_node((u, v), (w * dist, z, dist, s.clone()));
    
                // Not isolated; at least one neighbor found
                is_isolated = false;
            }
        }
        
        // If the source node is isolated, create a "synthetic" edge so it is connected somewhere
        if is_isolated {
            if let Some(&(_, z, ref s)) = node_mapping.get(&(i, j)) {
                // Distance to an adjacent cell in kilometers
                let (x2, y2) = transform.xy(j, i+1);
                let dist: f32 = distances::distance_km(x1, y1, x2, y2, is_wgs);
                // Calcualte the weight using max_cost
                let w: f32 = (1.0 - factor) * z + factor;
                // Make a fake node
                let fake_v = u + 1;
    
                // Clear the graph to ensure no isolated edge remains
                self.clear();
                self.add_node((u, fake_v), (w * dist, z, dist, s.clone()));
            }
        }
    
        is_isolated
    }
}


/// Process connections to the next level (e.g. from level 2 to level 4)
/// output graph: (u, v) (adj_cond, cond, dist, similarities)
/// u: source, v: destination
impl Graph {
    #[inline]
    pub fn fringe(
        &mut self,
        i: i32, j: i32,
        factor: f32,
        level: i32, 
        node_mapping: &HashMap<(i32, i32), (u32, f32, Arc<Vec<f32>>)>,
        node_mapping_higher: &HashMap<(i32, i32), (u32, f32, Arc<Vec<f32>>)>,
        transforms: &HashMap<i32, Affine>,
        is_wgs: bool,
    ) {
        if let Some(&(uu, _, _)) = node_mapping.get(&(i, j)) {
            let higher_level: i32 = level * 2;
            
            // Get all higher neighbours at once
            let higher_neighbours = get_edge_neighbours(i, j);

            // Get the Affines for distance calc
            let transform: &Affine = transforms.get(&level).unwrap();
            let transform_upper: &Affine = transforms.get(&higher_level).unwrap();
            // Get the actual coordinates values for distance calc
            let (x1, y1) = transform.xy(j, i);
            
            // Use 'ref' to borrow Vec<f32> rather than moving it
            for &(ni, nj) in &higher_neighbours {
                // Only if the neghbours are in the higher mapping proceess
                if let Some(&(v, z, ref s)) = node_mapping_higher.get(&(ni, nj)) {
                    let w = (1.0 - factor) * z + factor;

                    // Get the actual coordinates of the higher level
                    let (x2, y2) = transform_upper.xy(nj, ni);
                    // Distance in kilometer
                    let dist: f32 = distances::distance_km(x1, y1, x2, y2, is_wgs);
                            
                    self.add_node((uu, v), (w * dist, z, dist, s.clone()));
                }
            }
        }
    }
}


/// Get the 3 possible neighbours of edge cells in the higher level 
/// i.e. the link between a level edge to its higher level cells
fn get_edge_neighbours(i: i32, j: i32) -> [(i32, i32); 3] {
    // Higher level cell containing the target cell
    let target_higher = (i >> 1, j >> 1);    
    // 8 neighbor offsets: N, S, W, E, NW, NE, SW, SE
    const OFFSETS: [(i32, i32); 8] = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ];
    
    // Collect unique higher cells (exactly 3 after excluding target_higher)
    let mut higher_cells = [(0, 0); 3];
    let mut count = 0;
    
    for (di, dj) in OFFSETS {
        let ni = i + di;
        let nj = j + dj;
        let higher = (ni >> 1, nj >> 1);
        
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
        if !found && count < 3 {
            higher_cells[count] = higher;
            count += 1;
        }
    }
    
    higher_cells
}

