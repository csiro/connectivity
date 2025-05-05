use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::PyArray2;
use ndarray::Array2;

use std::collections::HashMap;
use ordered_float::NotNan;
use pathfinding::prelude::dijkstra_all;

mod window;
use window::multi_level_window;

mod graph;
use graph::multi_level_graph_optimized;


#[pyfunction]
fn py_multi_level_window(
    base_i: i32,
    base_j: i32,
    current_level: i32,
    data_dict: &PyAny,
    nb_size: i32,
    last_nb_size: i32,
) -> PyResult<(Vec<i32>, Vec<i32>, Vec<f32>)> {
    let mut rust_data_dict: HashMap<i32, Array2<f32>> = HashMap::new();

    // Get Python's items() method
    let items = data_dict.call_method0("items")?;
    // Iterate over the (key, value) pairs
    let iter = items.iter()?;
    
    for pair in iter {
        let (key_obj, value_obj) = pair?.extract::<(&PyAny, &PyAny)>()?;
        let key = key_obj.extract::<i32>()?;
        let py_array = value_obj.extract::<&PyArray2<f32>>()?;
        let array_owned = unsafe { py_array.as_array().to_owned() };
        rust_data_dict.insert(key, array_owned);
    }

    
    let mut level_dict: HashMap::<i32, (Vec<i32>, Vec<i32>, Vec<f32>)> = HashMap::new();

    for &key in rust_data_dict.keys() {
        level_dict.insert(
            key,
            multi_level_window(
                base_i,
                base_j,
                current_level,
                &rust_data_dict,
                nb_size,
                last_nb_size,
            ),
        );
    }

    let factor: f32 = 0.5;

    let (graph, source) = multi_level_graph_optimized(
        base_i,
        base_j,
        &level_dict, 
        factor
    );

    // println!("graph: {:?}", graph);

    // let mut graph: HashMap<u32, Vec<(u32, NotNan<f32>)>> = HashMap::new();

    // Create the successors function that uses our adjacency map
    let successors = |node: &u32| -> Vec<(u32, u32)> {
        // Return the neighbors of this node from our graph
        // If the node isn't in our graph, return an empty vector
        match graph.get(node) {
            Some(neighbors) => neighbors.clone(),
            None => Vec::new(),
        }
    };

    // Note: You may need to adapt the dijkstra_all function to work with f32 values
    let reachables: HashMap<u32, (u32, u32)> = dijkstra_all(&source, successors);
    
    // Instead, you could print specific results if needed
    println!("Found {} reachable nodes", reachables.len());
    println!("Found {:?}", reachables);


    Ok(multi_level_window(
        base_i,
        base_j,
        current_level,
        &rust_data_dict,
        nb_size,
        last_nb_size,
    ))

}

#[pymodule]
fn multires_connectivity(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_multi_level_window, _py)?)?;
    Ok(())
}
