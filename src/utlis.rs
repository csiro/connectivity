use pyo3::types::{PyAny, PyAnyMethods};
use pyo3::Bound;
use numpy::{PyArray2, PyArray3};
use ndarray::{Array3, Array2, Array1, s};
use std::collections::HashMap;
use pathfinding::prelude::dijkstra_all;
use rayon::ThreadPoolBuilder;
use std::sync::Once;


static INIT_RAYON: Once = Once::new();

/// Initiate rayon with a specified number of cores
pub fn init_rayon_internal(n_threads: Option<usize>) {
    INIT_RAYON.call_once(|| {
        let threads = match n_threads {
            Some(n) if n > 0 => n,
            _ => num_cpus::get(),
        };

        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("Failed to build global thread pool");
    });
}


/// Convert a Python dict-of-arrays into a Rust `HashMap<i32, Array2<f32>>`.
pub fn to_2d_map(data_dict: &Bound<PyAny>) -> HashMap<i32, Array2<f32>> {
    let mut rust_map = HashMap::new();

    // call .items() on the Python dict
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


// Convert Python object to a rust_hashmap
pub fn to_3d_map<'py>(data_dict: &Bound<PyAny>) -> HashMap<i32, Array3<f32>> {
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


/// Get the transgrid values for ij cell
pub fn get_current(trans_maps: &Vec<HashMap<i32, Array3<f32>>>, i: usize, j: usize) -> Array1<f32> {
    let trans_array = &trans_maps[0];

    if let Some(array3) = trans_array.get(&1) {
        if i < array3.shape()[1] && j < array3.shape()[2] {
            return array3.slice(s![.., i, j]).to_owned(); // returns Array1<f32>
        }
    }
    // Return empty array if anything fails
    Array1::zeros(0)
}


/// Convert edge list to adjacency list format to be injested by dijksta_all function via successor
/// Converts f32 weights to u32 by multiplying by 100 and rounding to be used in Dijkstra
fn to_adjacency(
    graph_temp: &HashMap<(u16, u16), (f32, f32, f32, Vec<f32>)>,
    weighted: bool,
) -> HashMap<u16, Vec<(u16, u32)>> {
    // First pass: count edges per source node
    let mut edge_counts: HashMap<u16, usize> = HashMap::new();
    for &(u, _) in graph_temp.keys() {
        *edge_counts.entry(u).or_insert(0) += 1;
    }

    // Initialize the adjacency list with pre-allocated space
    let mut adjacency: HashMap<u16, Vec<(u16, u32)>> = HashMap::with_capacity(edge_counts.len());
    for (&node, &count) in &edge_counts {
        adjacency.insert(node, Vec::with_capacity(count));
    }

    // Second pass: fill adjacency list; u32 to avoid integer overflow
    for (&(u, v), &(weighted_dist, _, unweighted_dist, _)) in graph_temp {
        let weight = if weighted {
            (weighted_dist * 100.0).round() as u32
        } else {
            // NOTE: fix this by * 100 then divided when intact distance is calcualted.. this ignores sub unit of crs
            unweighted_dist.round() as u32 // not multipled by 100 as its sum is directly used as intact-dist
        };

        if let Some(neighbors) = adjacency.get_mut(&u) {
            neighbors.push((v, weight));
        }
    }

    adjacency
}


/// Create the reachable path with dijkstra; weighted by condition or not
pub fn dijkstra(
    graph: &HashMap<(u16, u16), (f32, f32, f32, Vec<f32>)>, 
    source: u16,
    weighted: bool
) -> HashMap<u16, (u16, u32)> {
    // Convert the graph to suitable format for dijkstra
    let graph_int = to_adjacency(&graph, weighted);
    let successors = |node: &u16| -> Vec<(u16, u32)> {
        graph_int.get(node).cloned().unwrap_or_default()
    };

    // Calculate all reachable paths; the end nodes/segments
    dijkstra_all(&source, successors)
}


/// Return distance values and the condition/similarity of the last segment
pub fn path_distance(
    graph: &HashMap<(u16, u16), (f32, f32, f32, Vec<f32>)>,
    path: &[u16],
    dist_intact: f32
) -> (f32, f32, f32, Vec<f32>) {
    let mut dist_adjusted = 0.0;
    let mut last_condition = 0.0;
    let mut last_sims_ref: Option<&Vec<f32>> = None;

    for (from, to) in path.windows(2).map(|w| (w[0], w[1])) {
        if let Some(&(_, cond, dist, ref sims)) = graph.get(&(from, to)) {
            dist_adjusted += dist / (0.5 * cond + 0.5);
            last_condition = cond;
            last_sims_ref = Some(sims);
        }
    }

    // Clone only once at the end
    let last_sims = last_sims_ref.cloned().unwrap_or_default(); 

    (dist_adjusted, dist_intact, last_condition, last_sims)
}


// // A progress struct to track where there's a need to update window for a level
// pub struct Progress {
//     tracker: HashMap<i32, (i32, i32)>,
// }

// impl Progress {
//     pub fn new() -> Self {
//         Progress {
//             tracker: HashMap::new(),
//         }
//     }

//     pub fn update(&mut self, i: i32, j: i32, factor: i32) -> bool {
//         let mut updated = false;

//         // Get current position or initialize to (-1, -1)
//         let (mut current_i, mut current_j) = self.tracker.get(&factor).copied().unwrap_or((-1, -1));

//         // Calculate new positions
//         let new_i = i / (factor * 2);
//         let new_j = j / (factor * 2);

//         // Check if position changed
//         if new_i != current_i {
//             current_i = new_i;
//             updated = true;
//         }
//         if new_j != current_j {
//             current_j = new_j;
//             updated = true;
//         }

//         // Update tracker
//         self.tracker.insert(factor, (current_i, current_j));
//         updated
//     }
// }

