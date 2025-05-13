use pyo3::prelude::*;
use numpy::{PyArray2, PyArray3};
use ndarray::{Array2, Array3};
use std::collections::HashMap;
use std::f32::consts::E;


// Convert Python object to a rust_hashmap
pub fn to_2d_map<'py>(
    data_dict: &'py PyAny,
) -> HashMap<i32, Array2<f32>> {
    let mut rust_data_dict = HashMap::new();

    let items = data_dict.call_method0("items").unwrap();
    let iter = items.iter().unwrap();

    for pair in iter {
        let (key_obj, value_obj) = pair.unwrap().extract::<(&PyAny, &PyAny)>().unwrap();
        let key = key_obj.extract::<i32>().unwrap();
        let py_array = value_obj.extract::<&PyArray2<f32>>().unwrap();
        let array_owned = unsafe { py_array.as_array().to_owned() };
        rust_data_dict.insert(key, array_owned);
    }

    rust_data_dict
}


// Convert Python object to a rust_hashmap
pub fn to_3d_map<'py>(
    data_dict: &'py PyAny,
) -> HashMap<i32, Array3<f32>> {
    let mut rust_data_dict = HashMap::new();

    let items = data_dict.call_method0("items").unwrap();
    let iter = items.iter().unwrap();

    for pair in iter {
        let (key_obj, value_obj) = pair.unwrap().extract::<(&PyAny, &PyAny)>().unwrap();
        let key = key_obj.extract::<i32>().unwrap();
        let py_array = value_obj.extract::<&PyArray3<f32>>().unwrap();
        let array_owned = unsafe { py_array.as_array().to_owned() };
        rust_data_dict.insert(key, array_owned);
    }

    rust_data_dict
}



/// Convert edge list to adjacency list format to be injested by dijksta_all function via successor
/// Converts f32 weights to u32 by multiplying by 1,000,000 and rounding to be used in Dijkstra
pub fn convert_to_adjacency(graph_temp: &HashMap<(u32, u32), (f32, f32, f32)>) -> HashMap<u32, Vec<(u32, u32)>> {
    // First pass: count edges per node to allocate exact sizes
    let mut sizes: HashMap<u32, usize> = HashMap::new();
    for &(u, _) in graph_temp.keys() {
        *sizes.entry(u).or_insert(0) += 1;
    }
    
    // Create graph with pre-allocated vectors
    let mut graph: HashMap<u32, Vec<(u32, u32)>> = HashMap::with_capacity(sizes.len());
    for (&node, &size) in &sizes {
        graph.insert(node, Vec::with_capacity(size));
    }
    
    // Second pass: fill the graph with converted weights
    // Using the weighted_dist (first element of value tuple)
    for ((u, v), &(weighted_dist, _, _)) in graph_temp {
        if let Some(edges) = graph.get_mut(u) {
            // Convert f32 weight to u32 by scaling and rounding
            let int_val: u32 = (weighted_dist * 1_000_000.0).round() as u32;
            edges.push((*v, int_val));
        }
    }
    
    graph
}


/// Generate consecutive pairs from a slice of values
fn consecutive_pairs<T: Copy>(values: &[T]) -> Vec<(T, T)> {
    values.windows(2)
          .map(|window| (window[0], window[1]))
          .collect()
}

/// Calculate the connectedness of a path
pub fn path_connectedness(
    graph: &HashMap<(u32, u32), (f32, f32, f32)>,
    path: &[u32],
    dispersal: f32
) -> f32 {
    // Convert path to pairs for lookup
    let path_pairs = consecutive_pairs(path);
    
    let mut dsum: f32 = 0.0;
    let mut dsum_max: f32 = 0.0;
    let mut dist_land: f32 = 0.0;
    let mut dist_intact: f32 = 0.0;
    let denom_val: f32 = 283.465; // 5.785 * (50 - 1)
    
    for &(from, to) in &path_pairs {
        // Look up the edge in the graph
        if let Some(&(dw, cond, dist)) = graph.get(&(from, to)) {
            // Update distances
            dist_land += dist / ((0.5 * cond) + 0.5); // Permeability calculation
            dist_intact += dist; // Distance of intact cells
            
            let dist_land_lm = dist_land / dispersal;
            let dist_intact_lm = dist_intact / dispersal;
            
            let exp_term = dist_land_lm * dist_land_lm / denom_val;
            let exp_term_max = dist_intact_lm * dist_intact_lm / denom_val;
            
            dsum += E.powf(-exp_term) * cond;
            dsum_max += E.powf(-exp_term_max);
        }
    }
    
    if dsum_max > 0.0 {
        dsum / dsum_max
    } else {
        0.0
    }
}

