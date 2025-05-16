use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use std::collections::HashMap;
use numpy::{PyArray2, ToPyArray};
use ndarray::{Array3, Array2, Array1, ArrayView1, s};
use pathfinding::prelude::{dijkstra_all, build_path};
use rayon::prelude::*;
// local modules
mod window;
mod graph;
mod utlis;
use window::{multi_level_window, multi_level_window2};
use graph::multi_level_graph;
use utlis::{to_adjacency, path_connectedness, to_2d_map, to_3d_map};

#[pyfunction]
fn get_list_rust(
    list_of_dict: &PyAny
) -> PyResult<Py<PyArray2<f32>>> {

    let nrows = 10;
    let ncols = 10;

    
    // Convert Python list into a native Rust Vec<HashMap<i32, Array3<f32>>>
    let list = list_of_dict.downcast::<PyList>()?; // Convert PyAny to PyList
    let rust_maps: Vec<HashMap<i32, Array3<f32>>> = list
        .iter()
        .map(|item| {
            let dict = item.downcast::<PyDict>().unwrap(); // Handle errors properly in production
            to_3d_map(dict) // Your existing conversion function
        })
        .collect();

    for map in rust_maps {
        println!("Parsed map with {} keys", map.len());
    }


    
    // Initialize output with zeros
    let mut outarray = Array2::<f32>::zeros((nrows, ncols));

    for i in 0..nrows {
        for j in 0..ncols {
            outarray[[i, j]] = (i + j) as f32;
        }
    }

    // Convert the array back to Python with gil
    Python::with_gil(|py| {
        let pyarray = outarray.to_pyarray_bound(py);
        Ok(pyarray.unbind())
    })
}


fn get_focal(trans_maps: &Vec<HashMap<i32, Array3<f32>>>, i: usize, j: usize) -> Array1<f32> {
    let trans_array = &trans_maps[0];

    if let Some(array3) = trans_array.get(&1) {
        if i < array3.shape()[1] && j < array3.shape()[2] {
            return array3.slice(s![.., i, j]).to_owned(); // returns Array1<f32>
        }
    }

    // Return empty array if anything fails
    Array1::zeros(0)
}


#[pyfunction]
fn _connectivity(
    data_dict: &PyAny,
    trans_list: &PyAny,
    lambda_val: f32,
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

                    let mut level_dict: HashMap::<i32, (Vec<i32>, Vec<i32>, Vec<f32>, Vec<Vec<f32>>)> = HashMap::new();

                    let focal_val: Array1<f32> = get_focal(&trans_maps, i, j);

                    for &level in cond_map.keys() {
                        level_dict.insert(
                            level,
                            multi_level_window2(
                                i as i32,
                                j as i32,
                                level,
                                &cond_map,
                                nb_size,
                                last_nb_size,
                                &trans_maps,
                                &focal_val
                            ),
                        );
                    }

                    // Get the graph and source node for cell ij
                    let (edge_graph, source) = multi_level_graph(i as i32, j as i32, &level_dict, scale);
                    // Convert the graph to suitable format for dijkstra
                    let graph = to_adjacency(&edge_graph);
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
                        // if beri {
                        // }
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
    m.add_function(wrap_pyfunction!(get_list_rust, _py)?)?;
    Ok(())
}


    // if let Some(array) = trans_maps[0].get(&1) {
    //     let focal_val: ArrayView1<f32> = array.slice(s![.., 300, 300]);
    //     // use `focal_val` here
    //     let (a, b, c, d) = multi_level_window2(
    //         300,
    //         300,
    //         1,
    //         &cond_map,
    //         nb_size,
    //         last_nb_size,
    //         &trans_maps,
    //         focal_val
    //     );
    //     println!("i: {:?}", a);
    //     println!("j: {:?}", b);
    //     println!("cond: {:?}", c);
    //     println!("diss: {:?}", d);
    // } else {
    //     println!("No array was found in the list")
    // }

