use std::collections::{HashMap, HashSet};
use std::boxed::Box;
use crate::affine::Affine;
use crate::graph::Graph;


/// Build a graph strcut from a multi-level window data
/// struct { HashMap<(source, destination) (adj-dist, condtion, dist, sims)>, source-node }
impl Graph {
    #[inline]
    pub fn from_data(
        i_base: i32, 
        j_base: i32, 
        factor: f32,
        windows: &HashMap<i32, (Vec<i32>, Vec<i32>, Vec<f32>, Vec<Vec<Option<f32>>>)>, 
        transforms: &HashMap<i32, Affine>,
        geographic: bool,
    ) -> Self {
        // Pre-compute queen-case neighbour indices
        const COLS: [i32; 8] = [0, 1, 0, -1, 1, 1, -1, -1];
        const ROWS: [i32; 8] = [1, 0, -1, 0, 1, -1, 1, -1];
        
        // Get and sort levels
        let mut levels: Vec<i32> = windows.keys().cloned().collect();
        levels.sort_unstable(); // faster than sort()
        
        let max_level = *levels.last().unwrap_or(&1);
              
        // Pre-allocate with capacity for better performance
        let guess_size = windows.values().map(|(i, _, _, _)| i.len() * 8).sum::<usize>();
        let mut graph_temp = Graph::new(Some(guess_size));
        
        // Node mappings of the next level;
        let size_i = windows.get(&1).map(|(iv, _, _, _)| iv.len()).unwrap_or(36);
        let mut node_mapping_higher = HashMap::with_capacity(size_i);
        
        // Edge indices as HashSet for faster lookups
        let mut all_edge_indices: HashMap<i32, HashSet<(i32, i32)>> = HashMap::new();
        // Pre-compute edge indices for all levels
        for (&level, (i_array, j_array, _, _)) in windows {
            let edge_indices = get_edge_indices(i_array, j_array);
            all_edge_indices.insert(level, edge_indices);
        }
        
        // Process each level
        'outer: for (iter_level, &level) in levels.iter().enumerate() {
            // Only proceed if level is there..
            if let Some((i_array, j_array, values, sims)) = windows.get(&level) {
                let num_cell = i_array.len();
                let edge_indices = all_edge_indices.get(&level).expect("Level not found in edge indices.");
                
                let level_affine: &Affine = transforms.get(&level).expect("Missing level in Affine set.");
                
                // Generate or update node mappings
                let node_mapping = if iter_level == 0 {
                    // First level node mapping
                    let nm = create_node_mapping(i_array, j_array, values, sims, level);
                    // Find the base node index only once
                    if let Some((_, (u, _, _))) = nm.get_key_value(&(i_base, j_base)) {
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
                    if let Some((i_array2, j_array2, values2, sims2)) = windows.get(&higher_level) {
                        node_mapping_higher = create_node_mapping(i_array2, j_array2, values2, sims2, higher_level);
                    }
                }
                
                // Process all cells in a level and the higher neighbours of the edge
                for cell_idx in 0..num_cell {
                    let i = i_array[cell_idx];
                    let j = j_array[cell_idx];
                    let u = cell_idx as u32 + level as u32 * 1000; // unique identifier of the node
                    
                    // Process neighbors at the current level
                    // Modify the graph, and return true if cell is isolated
                    let was_isolated = graph_temp.neighbours(
                        i, j, u,
                        &COLS, &ROWS,
                        factor,
                        &node_mapping,
                        &level_affine,
                        geographic,
                    );
        
                    // For source nodes with no neighbours (isolated pixel, e.g tiny islands), add duplicated values
                    // and break the outer loop to avoid processing the rest of levels for the current graph
                    if was_isolated {
                        break 'outer;
                    }
                    
                    // Process connections to the next level (e.g. from level 2 to level 4)
                    if level < max_level && edge_indices.contains(&(i, j)) {
                        graph_temp.fringe(
                            i, j,
                            factor,
                            level,
                            &node_mapping,
                            &node_mapping_higher,
                            &transforms,
                            geographic,
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
fn create_node_mapping(
    i_array: &[i32],
    j_array: &[i32],
    values: &[f32],
    similarities: &[Vec<Option<f32>>],
    level: i32
) -> HashMap<(i32, i32), (u32, f32, Box<Vec<Option<f32>>>)> {
    let level_id = level as u32 * 1000;
    let num_sims = similarities.len();
    let mut mapping = HashMap::with_capacity(i_array.len());
    
    for (i, (&i_val, &j_val)) in i_array.iter().zip(j_array).enumerate() {
        // Make a vector for each ij cell containing cell's simiality of all scenarios
        let mut sim_vals = Vec::with_capacity(num_sims);
        // Get sim values of each index/cell and put in a vec (already double checked it)
        for sim_vec in similarities {
            sim_vals.push(sim_vec[i]);
        }      
        // Wrap it in Box so later clones are cheap
        let sim_vals = Box::new(sim_vals);

        mapping.insert((i_val, j_val), (i as u32 + level_id, values[i], sim_vals));
    }
    
    mapping
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
    i_arr.iter().zip(j_arr.iter())
        .filter(|(&i, &j)| i == i_min || i == i_max || j == j_min || j == j_max)
        .for_each(|(&i, &j)| { edge_set.insert((i, j)); });
    
    edge_set
}

