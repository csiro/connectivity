use pyo3::types::{PyAny, PyAnyMethods};
use pyo3::Bound;
use pyo3::PyResult;
use numpy::{PyArray2, PyArray3};
use ndarray::{Array3, Array2, Array1, s};
use std::collections::HashMap;
use std::boxed::Box;
use pathfinding::prelude::dijkstra_all;
// local module
use crate::affine::Affine;
use crate::graph::Graph;


/// Convert a Python dict of rasterio transforms into a Rust HashMap.
pub fn to_transform_map(data_dict: &Bound<PyAny>) -> HashMap<i32, Affine> {
    let mut rust_map = HashMap::new();

    // Convert to iterable
    let iter = data_dict
        .call_method0("items").unwrap()
        .iter().unwrap();

    for pair in iter {
        let (key_obj, value_obj) = pair.unwrap().extract::<(&PyAny, &PyAny)>().unwrap();
        let key = key_obj.extract::<i32>().unwrap();
        // Extract the 6 f64 values from the tuple/sequence
        let transform_tuple = value_obj.extract::<(f64, f64, f64, f64, f64, f64)>().unwrap();
        // Convert to Affile class
        let affine_obj: Affine = transform_tuple.into();
        rust_map.insert(key, affine_obj);
    }

    rust_map
}


/// Convert a Python 2D numpy array into an Option<Arr> that can be a Rust `Array2<f32>`.
pub fn to_array<'py>(data: &Bound<'py, PyAny>) -> PyResult<Option<Array2<f32>>> {
    if data.is_none() {
        Ok(None) // Python passed `None`
    } else {
        let py_array = data.extract::<&PyArray2<f32>>()?;
        let array_owned = unsafe { py_array.as_array().to_owned() };
        Ok(Some(array_owned))
    }
}


/// Convert a Python dict of 2D-arrays into a Rust HashMap<i32, Array2<f32>>.
pub fn to_2d_map(data_dict: &Bound<PyAny>) -> HashMap<i32, Array2<f32>> {
    let mut rust_map = HashMap::new();

    // Convert to iterable
    let iter = data_dict
        .call_method0("items").unwrap()
        .iter().unwrap();

    for pair in iter {
        let (key_obj, value_obj) = pair.unwrap().extract::<(&PyAny, &PyAny)>().unwrap();
        let key = key_obj.extract::<i32>().unwrap();
        let py_array = value_obj.extract::<&PyArray2<f32>>().unwrap();
        let array_owned = unsafe { py_array.as_array().to_owned() };
        rust_map.insert(key, array_owned);
    }

    rust_map
}


/// Convert a Python dict of 3D-arrays into a Rust HashMap<i32, Array3<f32>>.
pub fn to_3d_map<'py>(data_dict: &Bound<PyAny>) -> HashMap<i32, Array3<f32>> {
    let mut rust_map = HashMap::new();

    // Convert to iterable
    let iter = data_dict
        .call_method0("items").unwrap()
        .iter().unwrap();

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
/// Converts f32 weights to u32 by multiplying with 100 and rounding to be used in Dijkstra
/// HashMap<u, Vec<(v, adj_cond/cond)>>
/// u: source, v: destination
#[inline]
fn to_adjacency(
    graph: &Graph,
    weighted: bool,
) -> HashMap<u32, Vec<(u32, u32)>> {
    // First pass: count edges per source node
    let edge_counts = graph.count_edges();

    // Initialize the adjacency list with pre-allocated space
    let mut adjacency: HashMap<u32, Vec<(u32, u32)>> = HashMap::with_capacity(edge_counts.len());
    for (&node, &count) in &edge_counts {
        adjacency.insert(node, Vec::with_capacity(count));
    }

    // Second pass: fill adjacency list; u32 to avoid integer overflow
    for (&(u, v), &(weighted_dist, _, unweighted_dist, _)) in &graph.data {
        // This is needed in integers, so multiplied by 100 to get upto 2 digits precision
        let weight = if weighted {
            (weighted_dist * 100.0).round() as u32
        } else {
            (unweighted_dist * 100.0).round() as u32
        };

        if let Some(neighbors) = adjacency.get_mut(&u) {
            neighbors.push((v, weight));
        }
    }

    adjacency
}


/// Create the reachable path with dijkstra; weighted by condition or not
pub fn dijkstra(
    graph: &Graph,
    weighted: bool
) -> HashMap<u32, (u32, u32)> {
    // Convert the graph to suitable format for dijkstra
    let graph_int = to_adjacency(&graph, weighted);
    let successors = |node: &u32| -> Vec<(u32, u32)> {
        graph_int.get(node).cloned().unwrap_or_default()
    };

    // Calculate all reachable paths; the end nodes/segments
    dijkstra_all(&graph.source, successors)
}


/// Return distance values and the condition/similarity of the last segment
pub fn path_distance(
    graph: &Graph,
    path: &[u32],
    dist_intact: f32
) -> (f32, f32, f32, Box<Vec<f32>>) {
    let mut dist_adjusted = 0.0;
    let mut last_condition = 0.0;
    let mut last_sims = Box::new(Vec::new());

    for (from, to) in path.windows(2).map(|w| (w[0], w[1])) {
        if let Some(&(_, cond, dist, ref sims)) = graph.get(&(from, to)) {
            dist_adjusted += dist / (0.5 * cond + 0.5);
            last_condition = cond;
            last_sims = Box::clone(sims); // Cloning Box is cheap; just a counter to heap
        }
    }

    (dist_adjusted, dist_intact, last_condition, last_sims)
}

