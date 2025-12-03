use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3::Bound;
use std::sync::Arc;
use std::collections::HashMap;
use numpy::{PyArray2, ToPyArray};
use ndarray::{Array3, Array2, Array1};
use pathfinding::prelude::build_path;
use rayon::prelude::*;
// local modules
mod window;
mod builder;
mod utils;
mod metrics;
mod distances;
mod affine;
mod graph;
use affine::Affine;
use graph::Graph;


#[pyfunction(signature = (data_dict, trans_list, transforms, lambdas, is_geo, max_cost, window_size, outer_window, n_threads=None))]
fn connectivity(
    data_dict: &Bound<PyAny>,
    trans_list: &Bound<PyAny>,
    transforms: &Bound<PyAny>,
    lambdas: Vec<f32>,
    is_geo: bool,
    max_cost: f32,
    window_size: i32,
    outer_window: i32,
    n_threads: Option<usize>,
) -> PyResult<Py<PyArray2<f32>>> {

    // Create a Rust HashMap from Py data
    let cond_map: HashMap<i32, Array2<f32>> = utils::to_2d_map(data_dict);
    let num_levels = cond_map.len();

    // Convert Python list into a native Rust Vec<HashMap<i32, Array3<f32>>>
    let list = trans_list.downcast::<PyList>()?; // Convert PyAny to PyList
    let trans_maps: Vec<HashMap<i32, Array3<f32>>> = list.iter().map(|item| {
        let dict = item.downcast::<PyDict>().unwrap(); // handling errors properly
        utils::to_3d_map(dict)
    }).collect();

    // Convert the tranform of each level for coordinate and distance calcaulations
    let transform_map: HashMap<i32, Affine> = utils::to_transform_map(transforms);

    // If transgrids are provided run BERI, otherwise connectedness.
    let run_beri = !trans_maps.is_empty() && trans_maps.iter().any(|map| !map.is_empty());
  
    // Check condition dictionay was not empty and run the code for level 1 (original resolution)
    if let Some(array) = cond_map.get(&1) {
        let (nrows, ncols) = (array.shape()[0], array.shape()[1]);
        // Initialize output with zeros
        let mut outarray = Array2::<f32>::zeros((nrows, ncols));

        // Set the number of cores for parallel processing with Rayon
        let threads = match n_threads {
            Some(n) if n > 0 => n,
            _ => num_cpus::get(),
        };
        // Create an isolated custom thread pool as a local setting rather than global
        let custom_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("Failed to build thread pool");

        // Parallel iteration over rows in the thread pool
        let out_vec: Vec<(usize, Vec<f32>)> = custom_pool.install(|| {
            (0..nrows)
                .into_par_iter()
                .map(|i| {
                    let mut row_result = vec![f32::NAN; ncols];
                
                    for j in 0..ncols {
                        // Skip an NaN in the orginal resolution of the condition data
                        if array[[i, j]].is_nan() {
                            continue;
                        }

                        let mut windows: HashMap::<i32, (Vec<i32>, Vec<i32>, Vec<f32>, Vec<Vec<f32>>)> = HashMap::with_capacity(num_levels);
                        // Get the transgrid values for ij cell for the current climate
                        let ij_values: Array1<f32> = utils::get_current(&trans_maps, i, j);

                        for &level in cond_map.keys() {
                            let win = window::build_window(
                                i as i32,
                                j as i32,
                                level,
                                window_size,
                                outer_window,
                                &cond_map,
                                &trans_maps,
                                &ij_values,
                            );
                            windows.insert(level, win);
                        }

                        // Build a Graph for the cell ij
                        let the_graph = Graph::from_data(
                            i as i32, 
                            j as i32, 
                            max_cost,
                            &windows, 
                            &transform_map,
                            is_geo
                        );
                        // Calculate all reachable paths using weighted distance by conditon; altered condition
                        let nodes_altered  = utils::dijkstra(&the_graph, true);
                        // Using unweighted distance, i.e. intact condition case for the denominator
                        let nodes_intact = utils::dijkstra(&the_graph, false); 
                        
                        let mut cell_paths: Vec<(f32, f32, f32, Arc<Vec<f32>>)> = Vec::with_capacity(nodes_altered.len());

                        for &k in nodes_altered.keys() {
                            // Calcaulate optimal path for each reachable path
                            let optim_path = build_path(&k, &nodes_altered);
                            // Get the intact distance from source; divided by 100 to cancel out from path adjacency
                            let dist_intact: f32 = nodes_intact[&k].1 as f32 / 100.0;
                            // Get the path info for each target segment/node
                            cell_paths.push(utils::path_distance(&the_graph, &optim_path, dist_intact));
                        }

                        // Calculate BERI or Connectedness
                        row_result[j] = if lambdas.is_empty() {
                            0.0
                        } else {
                            let sum: f32 = lambdas.iter().map(|&lambda| {
                                if run_beri {
                                    metrics::beri_score(&cell_paths, lambda)
                                } else {
                                    metrics::connectedness(&cell_paths, lambda)
                                }
                            }).sum();
                            
                            sum / lambdas.len() as f32
                        };
                    }

                    (i, row_result)
                })
                .collect()
        });

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

