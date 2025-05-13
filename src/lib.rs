use pyo3::prelude::*;
use std::collections::HashMap;
use numpy::{PyArray2, ToPyArray};
use ndarray::{Array3, Array2, ArrayView1, s};
use pathfinding::prelude::{dijkstra_all, build_path};
use rayon::prelude::*;
// local modules
mod window;
mod graph;
mod utlis;
use window::{multi_level_window, multi_level_window2};
use graph::multi_level_graph;
use utlis::{convert_to_adjacency, path_connectedness, to_2d_map, to_3d_map};


#[pyfunction]
fn _connectivity(
    data_dict: &PyAny,
    trans_dict: &PyAny,
    lambda_val: f32,
    scale: f32,
    nb_size: i32,
    last_nb_size: i32,
) -> PyResult<Py<PyArray2<f32>>> {

    // Create a Rust HashMap from Py data
    let cond_map: HashMap<i32, Array2<f32>> = to_2d_map(data_dict);
    let tans_map: HashMap<i32, Array3<f32>> = to_3d_map(trans_dict);


    // if let Some(arr) = tans_map.get(&1) {
    //     let focal_val: ArrayView1<f32> = arr.slice(s![.., 300, 300]);
    //     println!(
    //         "MLW: {:?}", 
    //         multi_level_window2(
    //             300,
    //             300,
    //             1,
    //             &cond_map,
    //             nb_size,
    //             last_nb_size,
    //             &tans_map,
    //             focal_val
    //         )
    //     )
    // }

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

                for j in 0..ncols {

                    let mut level_dict: HashMap::<i32, (Vec<i32>, Vec<i32>, Vec<f32>)> = HashMap::new();

                    // let focal_val: ArrayView1<f32> = array.slice(s![.., i, j]);

                    for &key in cond_map.keys() {
                        level_dict.insert(
                            key,
                            multi_level_window(
                                i as i32,
                                j as i32,
                                key,
                                &cond_map,
                                nb_size,
                                last_nb_size,
                            ),
                        );
                    }

                    // Get the graph and source node for cell ij
                    let (edge_graph, source) = multi_level_graph(i as i32, j as i32, &level_dict, scale);
                    // Convert the graph to suitable format for dijkstra
                    let graph = convert_to_adjacency(&edge_graph);
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
                        let path_conn = path_connectedness(&edge_graph, &optim_path, lambda_val);
                        if path_conn > 0.0 {
                            len_paths += 1.0;
                        }
                        conn += path_conn;
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
fn rust_connectivity(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_connectivity, _py)?)?;
    Ok(())
}

