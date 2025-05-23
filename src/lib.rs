use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3::Bound;
use std::collections::HashMap;
use numpy::{PyArray2, ToPyArray};
use ndarray::{Array3, Array2, Array1};
use pathfinding::prelude::{dijkstra_all, build_path};
use rayon::prelude::*;
// local modules
mod window;
mod graph;
mod utlis;
mod metrics;
use metrics::{beri_score, connectedness};
use window::multi_level_window;
use graph::multi_level_graph;
use utlis::{*};


#[pyfunction]
fn connectivity(
    data_dict: &Bound<PyAny>,
    trans_list: &Bound<PyAny>,
    lambdas: Vec<f32>,
    scale: f32,
    nb_size: i32,
    last_nb_size: i32,
) -> PyResult<Py<PyArray2<f32>>> {

    // Create a Rust HashMap from Py data
    let cond_map: HashMap<i32, Array2<f32>> = to_2d_map(data_dict);

    // Convert Python list into a native Rust Vec<HashMap<i32, Array3<f32>>>
    let list = trans_list.downcast::<PyList>()?; // Convert PyAny to PyList
    let trans_maps: Vec<HashMap<i32, Array3<f32>>> = list
        .iter()
        .map(|item| {
            let dict = item.downcast::<PyDict>().unwrap(); // Handle errors properly in production
            to_3d_map(dict) // Your existing conversion function
        })
        .collect();

    // If transgrids are provided run BERI, otherwise connectedness.
    let run_beri = !trans_maps.is_empty() && trans_maps.iter().any(|map| !map.is_empty());

    // Check condition dictionay was not empty and run the code    
    if let Some(array) = cond_map.get(&1) {
        let shape = array.shape();
        let nrows = shape[0];
        let ncols = shape[1];

        // Initialize output with zeros
        let mut outarray = Array2::<f32>::zeros((nrows, ncols));

        // Parallel iteration over rows
        let out_vec: Vec<(usize, Vec<f32>)> = (0..nrows).into_par_iter()
            .map(|i| {
                let mut row_result = vec![0.0; ncols];
                // let mut progress = Progress::new();
                
                for j in 0..ncols {
                    
                    let mut level_dict: HashMap::<i32, (Vec<i32>, Vec<i32>, Vec<f32>, Vec<Vec<f32>>)> = HashMap::new();
                    let ij_values: Array1<f32> = get_values(&trans_maps, i, j);

                    for &level in cond_map.keys() {
                        // if progress.update(i as i32, j as i32, level) || j == 0 {
                            let window = multi_level_window(
                                i as i32,
                                j as i32,
                                level,
                                &cond_map,
                                nb_size,
                                last_nb_size,
                                &trans_maps,
                                &ij_values,
                            );
                            level_dict.insert(level, window);
                        // }
                    }

                    // Get the graph and source node for cell ij
                    let (edge_graph, source) = multi_level_graph(i as i32, j as i32, &level_dict, scale);
                    // Convert the graph to suitable format for dijkstra
                    let graph = to_adjacency(&edge_graph);
                    let successors = |node: &u16| -> Vec<(u16, u32)> {
                        graph.get(node).cloned().unwrap_or_default()
                    };

                    // Calculate all reachable paths; the end nodes
                    let reachables: HashMap<u16, (u16, u32)> = dijkstra_all(&source, successors);

                    let mut cell_paths: Vec<(f32, f32, f32, Vec<f32>)> = Vec::with_capacity(reachables.len());
                    // let mut cell_paths: Vec<(f32, f32, f32, Vec<f32>)> = Vec::new();

                    for &k in reachables.keys() {
                        // Calcaulate optimal path for each reachable path
                        let optim_path = build_path(&k, &reachables);
                        // Get the path info for each target segment/node
                        cell_paths.push(path_distance(&edge_graph, &optim_path));
                    }

                    // Calculate BERI or Connectedness
                    row_result[j] = if !lambdas.is_empty() {
                        if run_beri {
                            // Calculate BERI if transgrids are provided.
                            let sum: f32 = lambdas
                                .iter()
                                .map(|&lambda| beri_score(&cell_paths, lambda))
                                .sum();
                            sum / lambdas.len() as f32
                        } else {
                            // Calcualte connectedness
                            let sum: f32 = lambdas
                                .iter()
                                .map(|&lambda| connectedness(&cell_paths, lambda))
                                .sum();
                            sum / lambdas.len() as f32
                        }
                    } else {
                        0.0
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
fn rust_conn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction_bound!(connectivity, m)?)?;
    Ok(())
}

