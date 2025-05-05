use std::collections::HashMap;
use std::cmp;
use ordered_float::NotNan;

// Type definitions to match the Python types
type Int32Array = Vec<i32>;
type Float32Array = Vec<f32>;
type LevelDict = HashMap<i32, (Int32Array, Int32Array, Float32Array)>;
type NodeMapping = HashMap<(i32, i32), (u32, f32)>;
type EdgeIndices = HashMap<(i32, i32), bool>;
type GraphType = HashMap<(u32, u32), (f32, f32, f32)>; // (weighted_dist, z, dist)


/// Create a graph from multi-resolution raster array
/// The edge of the graph is habitat condition weighted by the distance between nodes
pub fn multi_level_graph(
    i_base: i32, 
    j_base: i32, 
    level_dict: &LevelDict, 
    factor: f32, 
    nb_size: i32
) -> (HashMap<u32, Vec<(u32, NotNan<f32>)>>, u32, Vec<u32>) {
    // Prepare the neighbor offsets
    let i_ngb = [0, 1, 0, -1, 1, 1, -1, -1];
    let j_ngb = [1, 0, -1, 0, 1, -1, 1, -1];
    
    // Intermediate graph representation matching the Python version
    let mut graph_temp: GraphType = HashMap::new();
    
    // Our final graph representation
    let mut graph: HashMap<u32, Vec<(u32, NotNan<f32>)>> = HashMap::new();
    
    // Get the sorted levels
    let mut levels: Vec<i32> = level_dict.keys().cloned().collect();
    levels.sort();
    
    let max_level = *levels.iter().max().unwrap();
    
    let mut source = 0;
    let mut targets = Vec::new();
    
    // Node mappings
    let mut node_mapping = NodeMapping::new();
    let mut node_mapping_higher = NodeMapping::new();
    
    for &level in &levels {
        let (i_array, j_array, values) = level_dict.get(&level).unwrap();
        let num_points = i_array.len();
        let edge_indices = get_edge_indices(i_array, j_array);
        
        if level > 1 {
            node_mapping = node_mapping_higher.clone();
        } else {
            node_mapping = create_node_mapping(i_array, j_array, values, level);
            
            // Find the base node index
            for (k, &val) in node_mapping.iter() {
                if k == &(i_base, j_base) {
                    source = val.0;
                    break;
                }
            }
        }
        
        if level < max_level {
            let higher_level = level * 2;
            if let Some((i_array2, j_array2, values2)) = level_dict.get(&higher_level) {
                node_mapping_higher = create_node_mapping(i_array2, j_array2, values2, higher_level);
            }
        }
        
        for point_idx in 0..num_points {
            let i = i_array[point_idx];
            let j = j_array[point_idx];
            let u = point_idx as u32 + level as u32 * 100;
            
            // Connect to neighbors at the same level
            for k in 0..8 {
                let ni = i + i_ngb[k];
                let nj = j + j_ngb[k];
                
                if let Some(&(v, z)) = node_mapping.get(&(ni, nj)) {
                    let dist = cell_distance(i, j, ni, nj) * level as f32;
                    // Conversion of condition value for graph
                    let w = (1.0 - factor) * z + factor;
                    
                    // Store edge in temp graph
                    graph_temp.insert((u, v), (w * dist, z, dist));
                }
            }
            
            // If not the max level, connect the nodes to higher level
            if level < max_level && edge_indices.contains_key(&(i, j)) {
                // Calculate the w/z of the u node to make the reverse connection
                if let Some(&(u_val, zz)) = node_mapping.get(&(i, j)) {
                    let wu = (1.0 - factor) * zz + factor;
                    
                    for (key, dist) in cell_higher_neighbours(i, j) {
                        // Check key/node is actually in the higher node_mapping
                        if let Some(&(v, z)) = node_mapping_higher.get(&key) {
                            // Conversion of condition value for graph
                            let w = (1.0 - factor) * z + factor;
                            
                            // Both-way edge so it can search back and for connectedness calc
                            graph_temp.insert((u_val, v), (w * dist, z, dist));
                            graph_temp.insert((v, u_val), (wu * dist, zz, dist));
                        }
                    }
                }
            } else if edge_indices.contains_key(&(i, j)) {
                targets.push(u);
            }
        }
    }
    
    // Convert the intermediate graph to the final format
    for ((u, v), (weighted_dist, _, _)) in graph_temp {
        let weight = NotNan::new(weighted_dist).unwrap();
        graph.entry(u).or_insert_with(Vec::new).push((v, weight));
    }
    
    (graph, source, targets)
}



/// Find the minimum or maximum value in an array of i32
fn minmax_int32(arr: &[i32], find_min: bool) -> i32 {
    if arr.is_empty() {
        return 0;
    }
    
    if find_min {
        *arr.iter().min().unwrap()
    } else {
        *arr.iter().max().unwrap()
    }
}

/// Get edge cells of each level
fn get_edge_indices(i_arr: &[i32], j_arr: &[i32]) -> EdgeIndices {
    let i_min = minmax_int32(i_arr, true);
    let i_max = minmax_int32(i_arr, false);
    let j_min = minmax_int32(j_arr, true);
    let j_max = minmax_int32(j_arr, false);
    let n = i_arr.len();
    
    let mut edge_dict = EdgeIndices::new();
    
    for i in 0..n {
        if i_arr[i] == i_min || i_arr[i] == i_max || j_arr[i] == j_min || j_arr[i] == j_max {
            edge_dict.insert((i_arr[i], j_arr[i]), true);
        }
    }
    
    edge_dict
}

/// Calculate distance between two cells
fn cell_distance(i1: i32, j1: i32, i2: i32, j2: i32) -> f32 {
    let di = i2 - i1;
    let dj = j2 - j1;
    ((di * di + dj * dj) as f32).sqrt()
}

/// For each edge cell, get its neighbor in the higher level including distances
fn cell_higher_neighbours(i: i32, j: i32) -> HashMap<(i32, i32), f32> {
    let i_shifted = i >> 1;  // Faster than i / 2
    let j_shifted = j >> 1;  // Faster than j / 2
    let x = j as f32 + 0.5;  // center x of the cell
    let y = i as f32 + 0.5;  // center y of the cell
    let higher_res = 2.0;
    
    // Pre-allocated arrays for better memory locality
    let neighbour_i = [-1, 1, 0, 0, -1, -1, 1, 1];
    let neighbour_j = [0, 0, -1, 1, -1, 1, -1, 1];
    
    let mut distances = HashMap::new();
    
    // Process all 8 neighbors
    for k in 0..8 {
        let ni = neighbour_i[k];
        let nj = neighbour_j[k];
        let ni_shifted = i_shifted + ni;
        let nj_shifted = j_shifted + nj;
        
        // Center coordinates of neighboring higher-level cells
        let x_higher = nj_shifted as f32 * higher_res + 1.0;
        let y_higher = ni_shifted as f32 * higher_res + 1.0;
        
        // Fast distance calculation
        let dx = x_higher - x;
        let dy = y_higher - y;
        let dist = (dx * dx + dy * dy).sqrt();
        
        distances.insert((ni_shifted, nj_shifted), dist);
    }
    
    distances
}

/// Create a node mapping from arrays of coordinates and values
fn create_node_mapping(
    i_array: &[i32], 
    j_array: &[i32], 
    values: &[f32], 
    level: i32
) -> NodeMapping {
    let level_id = level * 100;
    let mut mapping = NodeMapping::new();
    
    for i in 0..i_array.len() {
        let node = i as u32 + level_id as u32;
        mapping.insert((i_array[i], j_array[i]), (node, values[i]));
    }
    
    mapping
}
