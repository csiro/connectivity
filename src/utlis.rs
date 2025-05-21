use pyo3::prelude::*;
use pyo3::types::{PyAny};
use numpy::{PyArray2, PyArray3};
use ndarray::{Array3, Array2, Array1, s};
use std::collections::HashMap;
use std::f32::consts::E;


// Convert Python object to a rust_hashmap
pub fn to_2d_map<'py>(
    data_dict: &'py PyAny,
) -> HashMap<i32, Array2<f32>> {
    let mut rust_map = HashMap::new();

    let items = data_dict.call_method0("items").unwrap();
    let iter = items.iter().unwrap();

    for pair in iter {
        let (key_obj, value_obj) = pair.unwrap().extract::<(&PyAny, &PyAny)>().unwrap();
        let key = key_obj.extract::<i32>().unwrap();
        let py_array = value_obj.extract::<&PyArray2<f32>>().unwrap();
        let array_owned = unsafe { py_array.as_array().to_owned() };
        rust_map.insert(key, array_owned);
    }

    rust_map
}


// pub fn to_3d_map(dict: &PyDict) -> HashMap<i32, Array3<f32>> {
    //     let mut rust_map = HashMap::new();
    //     for (key, value) in dict {
        //         let k: i32 = key.extract()?;
//         let py_array: &PyArray3<f32> = value.downcast()?;
//         let arr = unsafe { py_array.as_array().to_owned() };
//         rust_map.insert(k, arr);
//     }
//     rust_map
// }

// Convert Python object to a rust_hashmap
pub fn to_3d_map<'py>(
    data_dict: &'py PyAny,
) -> HashMap<i32, Array3<f32>> {
    let mut rust_map = HashMap::new();

    let items = data_dict.call_method0("items").unwrap();
    let iter = items.iter().unwrap();

    for pair in iter {
        let (key_obj, value_obj) = pair.unwrap().extract::<(&PyAny, &PyAny)>().unwrap();
        let key = key_obj.extract::<i32>().unwrap();
        let py_array = value_obj.extract::<&PyArray3<f32>>().unwrap();
        let array_owned = unsafe { py_array.as_array().to_owned() };
        rust_map.insert(key, array_owned);
    }

    rust_map
}


/// Convert edge list to adjacency list format to be injested by dijksta_all function via successor
/// Converts f32 weights to u32 by multiplying by 1,000,000 and rounding to be used in Dijkstra
pub fn to_adjacency(graph_temp: &HashMap<(u32, u32), (f32, f32, f32, Vec<f32>)>) -> HashMap<u32, Vec<(u32, u32)>> {
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
    for ((u, v), &(weighted_dist, _, _, _)) in graph_temp {
        if let Some(edges) = graph.get_mut(u) {
            // Convert f32 weight to u32 by scaling and rounding
            let int_val: u32 = (weighted_dist * 1_000_000.0).round() as u32;
            edges.push((*v, int_val));
        }
    }
    
    graph
}


/// Get the transgrid values for ij cell
pub fn get_values(trans_maps: &Vec<HashMap<i32, Array3<f32>>>, i: usize, j: usize) -> Array1<f32> {
    let trans_array = &trans_maps[0];

    if let Some(array3) = trans_array.get(&1) {
        if i < array3.shape()[1] && j < array3.shape()[2] {
            return array3.slice(s![.., i, j]).to_owned(); // returns Array1<f32>
        }
    }

    // Return empty array if anything fails
    Array1::zeros(0)
}


/// Return distance values and the condition/similarity of the last segment
pub fn path_distance(
    graph: &HashMap<(u32, u32), (f32, f32, f32, Vec<f32>)>,
    path: &[u32]
) -> (f32, f32, f32, Vec<f32>) {
    let mut dist_adjusted = 0.0;
    let mut dist_intact = 0.0;
    let mut last_condition = 0.0;
    let mut last_sims_ref: Option<&Vec<f32>> = None;

    for (from, to) in path.windows(2).map(|w| (w[0], w[1])) {
        if let Some(&(_, cond, dist, ref sims)) = graph.get(&(from, to)) {
            dist_adjusted += dist / (0.5 * cond + 0.5);
            dist_intact += dist;
            last_condition = cond;
            last_sims_ref = Some(sims);
        }
    }

    // Clone only once at the end
    let last_sims = last_sims_ref.cloned().unwrap_or_default(); 

    (dist_adjusted, dist_intact, last_condition, last_sims)
}




/// Aggregating senarios
#[inline]
fn minimax(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    
    // Find the minimum value
    let min = x.iter()
        .fold(f32::INFINITY, |acc, &val| acc.min(val));
    
    0.5 * (mean + min)
}


/// Compute a BERI score from a segment and a lambda
pub fn beri_score(segment: &[(f32, f32, f32, Vec<f32>)], lambda: f32) -> f32 {
    const DENOM_VAL: f32 = 283.465; // 5.785 * (50 - 1)
    
    if segment.is_empty() {
        return 0.0;
    }

    // Get number of scenarios (including current)
    let n_scenario = segment[0].3.len();
    if n_scenario == 0 {
        return 0.0;
    }
    
    // Initialize numerator vector with capacity
    let mut numerator = vec![0.0f32; n_scenario];
    let mut denominator = 0.0f32;

    // Process each segment
    for (dist_adj, dist, cond, similarities) in segment {
        // Skip invalid data points
        if similarities.len() < n_scenario {
            continue;
        }
        
        // Calculate numerator weight with dist_adj
        let dist_lambda_num = dist_adj / lambda;
        let exp_term_num = dist_lambda_num * dist_lambda_num / DENOM_VAL;
        let weight_num = E.powf(-exp_term_num) * cond;
        
        // Update numerator values with similarity of scenarios
        for (i, &sim) in similarities.iter().take(n_scenario).enumerate() {
            numerator[i] += weight_num * sim;
        }
        
        // Calculate denominator weight with dist
        let dist_lambda_denom = dist / lambda;
        let exp_term_denom = dist_lambda_denom * dist_lambda_denom / DENOM_VAL;
        let weight_denom = E.powf(-exp_term_denom);
        
        // Update denominator (using first similarity value)
        denominator += weight_denom * similarities[0];
    }
    
    if denominator > 0.0 {
        minimax(&numerator) / denominator
    } else {
        0.0
    }
}


// Calculate the connectedness of a path
pub fn connectedness(segment: &[(f32, f32, f32, Vec<f32>)], lambda: f32) -> f32 {
    let sum_conn: f32 = segment
        .iter()
        .map(|(dist_adj, dist, condition, _)| {
            let numerator = E.powf(- (dist_adj / lambda)) * condition;
            let denominator = E.powf(- (dist / lambda));
            if denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            }
        })
        .sum();

    let len_conn: f32 = segment.len() as f32;

    if len_conn > 0.0 {
        sum_conn / len_conn
    } else {
        0.0
    }
}

