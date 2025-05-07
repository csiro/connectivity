use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::PyArray2;
use ndarray::Array2;
use numpy::IntoPyArray;
use numpy::ToPyArray;

use std::f32::consts::E;
use std::collections::HashMap;
use pathfinding::prelude::dijkstra_all;
use pathfinding::prelude::build_path;

mod window;
use window::multi_level_window;
mod graph;
use graph::{multi_level_graph, convert_to_adjacency_list};


use rayon::prelude::*;
use ndarray::s; // For slicing



/// Generate consecutive pairs from a slice of values
fn consecutive_pairs<T: Copy>(values: &[T]) -> Vec<(T, T)> {
    values.windows(2)
          .map(|window| (window[0], window[1]))
          .collect()
}

/// Calculate the connectedness of a path
fn path_connectedness(
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
            dist_land += dist / ((0.5 * cond) + 0.5); // permeability calculation
            dist_intact += dist; // distance of intact cells
            
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



// #[pyfunction]
// fn connectivity(
//     data_dict: &PyAny,
//     lambda_val: f32,
//     scale: f32,
//     nb_size: i32,
//     last_nb_size: i32,
// ) -> PyResult<Py<PyArray2<f32>>> {
//     let mut rust_data_dict: HashMap<i32, Array2<f32>> = HashMap::new();

//     // Get Python's items() method
//     let items = data_dict.call_method0("items")?;
//     // Iterate over the (key, value) pairs
//     let iter = items.iter()?;
    
//     for pair in iter {
//         let (key_obj, value_obj) = pair?.extract::<(&PyAny, &PyAny)>()?;
//         let key = key_obj.extract::<i32>()?;
//         let py_array = value_obj.extract::<&PyArray2<f32>>()?;
//         let array_owned = unsafe { py_array.as_array().to_owned() };
//         rust_data_dict.insert(key, array_owned);
//     }

//     // Get the first array to determine dimensions
//     if let Some(array) = rust_data_dict.get(&1) {
//         let shape = array.shape();
//         let nrows = shape[0];
//         let ncols = shape[1];
    
//         // Create a new array with the same shape
//         let mut outarray = Array2::<f32>::zeros((nrows, ncols));

//         for i in 0..nrows {
//             for j in 0..ncols {
//                 let mut level_dict: HashMap::<i32, (Vec<i32>, Vec<i32>, Vec<f32>)> = HashMap::new();
            
//                 for &key in rust_data_dict.keys() {
//                     level_dict.insert(
//                         key,
//                         multi_level_window(
//                             i as i32,
//                             j as i32,
//                             key,  // Using key instead of current_level
//                             &rust_data_dict,
//                             nb_size,
//                             last_nb_size,
//                         ),
//                     );
//                 }
            
//                 let (edge_graph, source) = multi_level_graph(
//                     i as i32,
//                     j as i32,
//                     &level_dict, 
//                     scale
//                 );
            
//                 // Convert to adjacency list
//                 let graph = convert_to_adjacency_list(&edge_graph);
            
//                 // Create the successors function for dijkstra
//                 let successors = |node: &u32| -> Vec<(u32, u32)> {
//                     match graph.get(node) {
//                         Some(neighbors) => neighbors.clone(),
//                         None => Vec::new(),
//                     }
//                 };
            
//                 let reachables: HashMap<u32, (u32, u32)> = dijkstra_all(&source, successors);
            
//                 let mut conn: f32 = 0.0;
//                 let mut len_paths: f32 = 0.0;
                
//                 for &k in reachables.keys() {
//                     let optim_path = build_path(&k, &reachables);
//                     let connval = path_connectedness(&edge_graph, &optim_path, lambda_val);
//                     if connval > 0.0 {
//                         len_paths += 1.0;
//                     }
//                     conn += connval;
//                 }

//                 outarray[[i, j]] = if len_paths > 0.0 {
//                     conn / len_paths
//                 } else {
//                     f32::NAN
//                 };
//             }
//         }

//         Python::with_gil(|py| {
//             let pyarray = outarray.to_pyarray_bound(py); // safer binding
//             Ok(pyarray.unbind())
//         })
        
//     } else {
//         Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>("No array found with key=1"))
//     }
// }


#[pyfunction]
fn connectivity(
    data_dict: &PyAny,
    lambda_val: f32,
    scale: f32,
    nb_size: i32,
    last_nb_size: i32,
) -> PyResult<Py<PyArray2<f32>>> {
    let mut rust_data_dict: HashMap<i32, Array2<f32>> = HashMap::new();

    let items = data_dict.call_method0("items")?;
    let iter = items.iter()?;

    for pair in iter {
        let (key_obj, value_obj) = pair?.extract::<(&PyAny, &PyAny)>()?;
        let key = key_obj.extract::<i32>()?;
        let py_array = value_obj.extract::<&PyArray2<f32>>()?;
        let array_owned = unsafe { py_array.as_array().to_owned() };
        rust_data_dict.insert(key, array_owned);
    }

    if let Some(array) = rust_data_dict.get(&1) {
        let shape = array.shape();
        let nrows = shape[0];
        let ncols = shape[1];

        // Initialize output with zeros
        let mut outarray = Array2::<f32>::zeros((nrows, ncols));

        // Parallel iteration over rows
        let out_vec: Vec<(usize, Vec<f32>)> = (0..nrows).into_par_iter()
            .map(|i| {
                let mut row_result = vec![0.0; ncols];

                for j in 0..ncols {
                    let mut level_dict: HashMap::<i32, (Vec<i32>, Vec<i32>, Vec<f32>)> = HashMap::new();

                    for &key in rust_data_dict.keys() {
                        level_dict.insert(
                            key,
                            multi_level_window(
                                i as i32,
                                j as i32,
                                key,
                                &rust_data_dict,
                                nb_size,
                                last_nb_size,
                            ),
                        );
                    }

                    let (edge_graph, source) = multi_level_graph(i as i32, j as i32, &level_dict, scale);
                    let graph = convert_to_adjacency_list(&edge_graph);
                    let successors = |node: &u32| -> Vec<(u32, u32)> {
                        graph.get(node).cloned().unwrap_or_default()
                    };
                    let reachables: HashMap<u32, (u32, u32)> = dijkstra_all(&source, successors);

                    let mut conn: f32 = 0.0;
                    let mut len_paths: f32 = 0.0;

                    for &k in reachables.keys() {
                        let optim_path = build_path(&k, &reachables);
                        let connval = path_connectedness(&edge_graph, &optim_path, lambda_val);
                        if connval > 0.0 {
                            len_paths += 1.0;
                        }
                        conn += connval;
                    }

                    row_result[j] = if len_paths > 0.0 {
                        conn / len_paths
                    } else {
                        f32::NAN
                    };
                }

                (i, row_result)
            })
            .collect();

        // Write back results into outarray
        for (i, row) in out_vec {
            for (j, val) in row.into_iter().enumerate() {
                outarray[[i, j]] = val;
            }
        }

        Python::with_gil(|py| {
            let pyarray = outarray.to_pyarray_bound(py);
            Ok(pyarray.unbind())
        })
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>("No array found with key=1"))
    }
}


#[pymodule]
fn multires_connectivity(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(connectivity, _py)?)?;
    Ok(())
}

