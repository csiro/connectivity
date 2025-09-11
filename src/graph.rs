use std::collections::{HashMap, HashSet};


/// Optimized implementation of multi-level graph generation
pub fn multi_level_graph(
    i_base: i32, 
    j_base: i32, 
    level_dict: &HashMap<i32, (Vec<i32>, Vec<i32>, Vec<f32>, Vec<Vec<f32>>)>, 
    factor: f32
) -> (HashMap<(u16, u16), (f32, f32, f32, Vec<f32>)>, u16) {
    // Pre-compute constants
    let i_ngb = [0, 1, 0, -1, 1, 1, -1, -1];
    let j_ngb = [1, 0, -1, 0, 1, -1, 1, -1];
    
    // Get and sort levels
    let mut levels: Vec<i32> = level_dict.keys().cloned().collect();
    levels.sort_unstable(); // faster than sort()
    
    let max_level = *levels.last().unwrap_or(&0);
    
    // Pre-allocate with capacity for better performance
    let mut graph_temp: HashMap<(u16, u16), (f32, f32, f32, Vec<f32>)> = HashMap::with_capacity(
        level_dict.values().map(|(i, _, _, _)| i.len() * 8).sum()
    );
        
    let mut source = 0;
    // let mut targets = Vec::new();
    
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
    for (_level_idx, &level) in levels.iter().enumerate() {
        let (i_array, j_array, values, sims) = &level_dict[&level];
        let num_points = i_array.len();
        let edge_indices = &all_edge_indices[&level];
        
        // Update node mappings
        if level > 1 {
            // Recycle the node mapping already calculated for the hihger level 
            node_mapping = node_mapping_higher.clone();
        } else {
            node_mapping = create_node_mapping(i_array, j_array, values, sims, level);
            
            // Find the base node index - do it once only
            for (k, val) in &node_mapping {
                if *k == (i_base, j_base) {
                    source = val.0;  // `val` is a reference to a tuple, so `val.0` copies u32 (which is Copy)
                    break;
                }
            }
        }
        
        // Pre-compute higher level node mapping if needed
        if level < max_level {
            let higher_level = level * 2;
            if let Some((i_array2, j_array2, values2, sims2)) = level_dict.get(&higher_level) {
                node_mapping_higher = create_node_mapping(i_array2, j_array2, values2, sims2, higher_level);
            }
        }
        
        // Process points - this could be parallelized for large datasets
        for point_idx in 0..num_points {
            let i = i_array[point_idx];
            let j = j_array[point_idx];
            let u = point_idx as u16 + level as u16 * 100;
            
            // Process neighbors at current level
            process_current_level_neighbors(
                i, j, u, level,
                &i_ngb, &j_ngb,
                &node_mapping,
                factor,
                &mut graph_temp
            );
            
            // Process connections to higher level if needed
            if level < max_level && edge_indices.contains(&(i, j)) {
                process_higher_level_connections(
                    i, j,
                    &node_mapping,
                    &node_mapping_higher,
                    factor,
                    &mut graph_temp
                );
            }
        }
    }

    (graph_temp, source)
}


/// Create node mapping efficiently
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
#[inline]
fn process_current_level_neighbors(
    i: i32, j: i32, 
    u: u16, level: i32,
    i_ngb: &[i32], j_ngb: &[i32],
    node_mapping: &HashMap<(i32, i32), (u16, f32, Vec<f32>)>,
    factor: f32,
    graph_temp: &mut HashMap<(u16, u16), (f32, f32, f32, Vec<f32>)>
) {
    for k in 0..8 {
        let ni = i + i_ngb[k];
        let nj = j + j_ngb[k];
        // Use 'ref' to borrow Vec<f32> rather than moving it
        if let Some(&(v, z, ref s)) = node_mapping.get(&(ni, nj)) {
            let dist = cell_distance(i, j, ni, nj) * level as f32;
            let w = (1.0 - factor) * z + factor;
            
            // Store only the weighted distance in the temp graph
            graph_temp.insert((u, v), (w * dist, z, dist, s.clone()));
        }
    }
}


/// Process connections to higher level
/// output graph: (u, v) (adj_cond, cond, dist, similarities)
#[inline]
fn process_higher_level_connections(
    i: i32, j: i32,
    node_mapping: &HashMap<(i32, i32), (u16, f32, Vec<f32>)>,
    node_mapping_higher: &HashMap<(i32, i32), (u16, f32, Vec<f32>)>,
    factor: f32,
    graph_temp: &mut HashMap<(u16, u16), (f32, f32, f32, Vec<f32>)>
) {
    if let Some(&(u_val, zz, ref ss)) = node_mapping.get(&(i, j)) {
        let wu = (1.0 - factor) * zz + factor;
        
        // Get all higher neighbors at once
        let higher_neighbors = get_higher_neighbors(i, j);
        
        // Use 'ref' to borrow Vec<f32> rather than moving it
        for &(ni, nj, dist) in &higher_neighbors {
            if let Some(&(v, z, ref s)) = node_mapping_higher.get(&(ni, nj)) {
                let w = (1.0 - factor) * z + factor;
                
                // Both-way edge
                graph_temp.insert((u_val, v), (w * dist, z, dist, s.clone()));
                graph_temp.insert((v, u_val), (wu * dist, zz, dist, ss.clone()));
            }
        }
    }
}


/// Calculate distance between two cells
#[inline]
fn cell_distance<T>(i1: T, j1: T, i2: T, j2: T) -> f32
where
    T: Copy + Into<f64>,
{

    // NOTE: This must get XY coords only 

    // if is_latlong {
    //     dist_meters = distance_haversine(i1, j1, i2, j2)
    //     (dist_meters / 1000.0) as f32
    // } else {
    //
    let di = i2.into() - i1.into();
    let dj = j2.into() - j1.into();
    let dist_meters = (di * di + dj * dj).sqrt(); // in same units as indices
    // (dist_meters * resolution / 1000.0) as f32 // kilometres as f32
    dist_meters as f32
    // }
}


/// Pre-compute all higher neighbor distances to avoid redundant calculations
#[inline]
fn get_higher_neighbors(i: i32, j: i32) -> [(i32, i32, f32); 8] {
    let i_shifted = i >> 1;
    let j_shifted = j >> 1;
    let x = j as f32 + 0.5;
    let y = i as f32 + 0.5;
    let higher_res = 2.0;
    
    let neighbour_i = [-1, 1, 0, 0, -1, -1, 1, 1];
    let neighbour_j = [0, 0, -1, 1, -1, 1, -1, 1];
    
    let mut results = [(0, 0, 0.0); 8];
    
    for k in 0..8 {
        let ni = neighbour_i[k];
        let nj = neighbour_j[k];
        let ni_shifted = i_shifted + ni;
        let nj_shifted = j_shifted + nj;
        
        let x_higher = nj_shifted as f32 * higher_res + 1.0;
        let y_higher = ni_shifted as f32 * higher_res + 1.0;
        
        let dist = cell_distance(x, y, x_higher, y_higher);
        
        results[k] = (ni_shifted, nj_shifted, dist);
    }
    
    results
}


/// Get edge cells efficiently using HashSet
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

