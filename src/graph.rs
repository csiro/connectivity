use std::collections::{HashMap, HashSet};
use crate::affine::Affine;
use crate::distances;


/// A graph strcuture of the multi-level window data
/// HashMap<(source, destination) (adj-dist, condtion, dist, sims)>, source-node
pub fn multi_level_graph(
    i_base: i32, 
    j_base: i32, 
    level_dict: &HashMap<i32, (Vec<i32>, Vec<i32>, Vec<f32>, Vec<Vec<f32>>)>, 
    factor: f32,
    transforms: &HashMap<i32, Affine>,
    geographic: bool,
) -> (HashMap<(u16, u16), (f32, f32, f32, Vec<f32>)>, u16) {
    // Pre-compute constants
    const COLS: [i32; 8] = [0, 1, 0, -1, 1, 1, -1, -1];
    const ROWS: [i32; 8] = [1, 0, -1, 0, 1, -1, 1, -1];
    
    // Get and sort levels
    let mut levels: Vec<i32> = level_dict.keys().cloned().collect();
    levels.sort_unstable(); // faster than sort()
    
    let max_level = *levels.last().unwrap_or(&0);
    
    // Pre-allocate with capacity for better performance
    let mut graph_temp: HashMap<(u16, u16), (f32, f32, f32, Vec<f32>)> = HashMap::with_capacity(
        level_dict.values().map(|(i, _, _, _)| i.len() * 8).sum()
    );
    
    let mut source: u16 = 0;
    
    // Node mappings - pre-compute sizes for better allocation
    let mut node_mapping = HashMap::new();
    let mut node_mapping_higher = HashMap::new();
    
    // Edge indices as HashSet for faster lookups
    let mut all_edge_indices: HashMap<i32, HashSet<(i32, i32)>> = HashMap::new();
    // Pre-compute edge indices for all levels
    for (&level, (i_array, j_array, _, _)) in level_dict {
        let edge_indices = get_edge_indices(i_array, j_array);
        all_edge_indices.insert(level, edge_indices);
    }

    // Process each level
    for (iter_level, &level) in levels.iter().enumerate() {
        let (i_array, j_array, values, sims) = &level_dict[&level];
        let num_points = i_array.len();
        let edge_indices = &all_edge_indices[&level];

        let level_affine: &Affine = transforms.get(&level).unwrap();
        
        // Update node mappings
        if iter_level < 1 {
            // This branch runs only once; for the first level
            node_mapping = create_node_mapping(i_array, j_array, values, sims, level);
            // Find the base node index only once
            if let Some((_, (u, _, _))) = node_mapping.get_key_value(&(i_base, j_base)) {
                source = *u;
            }
        } else {
            // Recycle the node mapping already calculated for the hihger level from previous round
            node_mapping = std::mem::take(&mut node_mapping_higher);
        }
        
        // Pre-compute higher level node mapping if needed
        if level < max_level {
            let higher_level = level * 2;
            if let Some((i_array2, j_array2, values2, sims2)) = level_dict.get(&higher_level) {
                node_mapping_higher = create_node_mapping(i_array2, j_array2, values2, sims2, higher_level);
            }
        }
        
        // Process points
        for point_idx in 0..num_points {
            let i = i_array[point_idx];
            let j = j_array[point_idx];
            let u = point_idx as u16 + level as u16 * 100;
            
            // Process neighbors at current level
            current_level_neighbors(
                i, j, u,
                &COLS, &ROWS,
                &node_mapping,
                factor,
                &mut graph_temp,
                &level_affine,
                geographic,
            );
            
            // Process connections to higher level if needed
            if level < max_level && edge_indices.contains(&(i, j)) {
                higher_level_connections(
                    i, j,
                    &node_mapping,
                    &node_mapping_higher,
                    factor,
                    &mut graph_temp,
                    level,
                    &transforms,
                    geographic,
                );
            }
        }
    }

    (graph_temp, source)
}


/// Create node mapping (the unique ID of each node)
fn create_node_mapping(
    i_array: &[i32],
    j_array: &[i32],
    values: &[f32],
    similarities: &[Vec<f32>],
    level: i32
) -> HashMap<(i32, i32), (u16, f32, Vec<f32>)> {
    let level_id = level * 100;
    let mut mapping = HashMap::with_capacity(i_array.len());
    
    for i in 0..i_array.len() {
        // collect the i-th value from each similarity vector
        let sim_vals: Vec<f32> = similarities.iter().map(|v| v[i]).collect();
        mapping.insert(
            (i_array[i], j_array[i]),
            (i as u16 + level_id as u16, values[i], sim_vals)
        );
    }
    
    mapping
}


/// Process neighbors at current level
/// output graph: (u, v) (adj_cond, cond, dist, similarities)
/// u: source, v: destination
#[inline]
fn current_level_neighbors(
    i: i32, 
    j: i32, 
    u: u16, // source node
    i_ngb: &[i32], 
    j_ngb: &[i32],
    node_mapping: &HashMap<(i32, i32), (u16, f32, Vec<f32>)>,
    factor: f32,
    graph_temp: &mut HashMap<(u16, u16), (f32, f32, f32, Vec<f32>)>,
    transform: &Affine,
    is_wgs: bool,
) {
    for k in 0..8 {
        let ni = i + i_ngb[k];
        let nj = j + j_ngb[k];

        // Get XY coords from IJ and transfrom object
        let (x1, y1) = transform.xy(j, i);
        let (x2, y2) = transform.xy(nj, ni);

        // Use 'ref' to borrow Vec<f32> rather than moving it
        if let Some(&(v, z, ref s)) = node_mapping.get(&(ni, nj)) {
            // Distance in kilometer
            let dist: f32 = distances::distance_km(x1, y1, x2, y2, is_wgs);
            let w: f32 = (1.0 - factor) * z + factor;
            
            // Store only the weighted distance in the temp graph
            graph_temp.insert((u, v), (w * dist, z, dist, s.clone()));
        }
    }
}


/// Process connections to higher level (e.g. level 2 to level 4)
/// output graph: (u, v) (adj_cond, cond, dist, similarities)
/// u: source, v: destination
#[inline]
fn higher_level_connections(
    i: i32, j: i32,
    node_mapping: &HashMap<(i32, i32), (u16, f32, Vec<f32>)>,
    node_mapping_higher: &HashMap<(i32, i32), (u16, f32, Vec<f32>)>,
    factor: f32,
    graph_temp: &mut HashMap<(u16, u16), (f32, f32, f32, Vec<f32>)>,
    level: i32, 
    transforms: &HashMap<i32, Affine>,
    is_wgs: bool,
) {
    if let Some(&(uu, _, _)) = node_mapping.get(&(i, j)) {
        let higher_level: i32 = level * 2;
        
        // Get all higher neighbors at once
        let higher_neighbors = get_edge_neighbors(i, j);

        // Get the Affines for distance calc
        let transform: &Affine = transforms.get(&level).unwrap();
        let transform_upper: &Affine = transforms.get(&higher_level).unwrap();
        // Get the actual coordinates values for distance calc
        let (x1, y1) = transform.xy(j, i);
        
        // Use 'ref' to borrow Vec<f32> rather than moving it
        for &(ni, nj) in &higher_neighbors {
            // Only if the neghbours are in the higher mapping proceess
            if let Some(&(v, z, ref s)) = node_mapping_higher.get(&(ni, nj)) {
                let w = (1.0 - factor) * z + factor;

                // Get the actual coordinates of the higher level
                let (x2, y2) = transform_upper.xy(nj, ni);
                // Distance in kilometer
                let dist: f32 = distances::distance_km(x1, y1, x2, y2, is_wgs);
                        
                graph_temp.insert((uu, v), (w * dist, z, dist, s.clone()));
            }
        }
    }
}


/// Get the 3 possible neighbours of edge cells in the higher level 
/// i.e. the link between a level edge to its higher level cells
fn get_edge_neighbors(i: i32, j: i32) -> [(i32, i32); 3] {
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

