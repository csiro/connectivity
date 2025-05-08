use pyo3::prelude::*;
use std::collections::HashMap;
use numpy::{PyArray2, ToPyArray};
use ndarray::{Array2, s};
use pathfinding::prelude::{dijkstra_all, build_path};
use rayon::prelude::*;
// local modules
mod window;
mod graph;
mod utlis;
use window::multi_level_window;
use graph::multi_level_graph;
use utlis::{convert_to_adjacency_list, path_connectedness};


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

                    // Get the graph and source node for cell ij
                    let (edge_graph, source) = multi_level_graph(i as i32, j as i32, &level_dict, scale);
                    // Convert the graph to suitable format for dijkstra
                    let graph = convert_to_adjacency_list(&edge_graph);
                    let successors = |node: &u32| -> Vec<(u32, u32)> {
                        graph.get(node).cloned().unwrap_or_default()
                    };

                    // Calculate all reachable paths; the end nodes
                    let reachables: HashMap<u32, (u32, u32)> = dijkstra_all(&source, successors);

                    let mut conn: f32 = 0.0;
                    let mut len_paths: f32 = 0.0;

                    // Loop through all reachable paths and calcaulate connectivity
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

        // Convert the array back to Python with gil
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

