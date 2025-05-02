use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::PyArray2;
use ndarray::Array2;
use std::collections::HashMap;

// Create a module for the window functions
mod window;
use window::multi_level_window;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// Python wrapper for the multi_level_window function
#[pyfunction]
fn py_multi_level_window(
    py: Python<'_>,
    base_i: i32,
    base_j: i32,
    current_level: i32,
    data_dict: Py<PyDict>,
    nb_size: i32,
    last_nb_size: i32,
) -> PyResult<(Vec<i32>, Vec<i32>, Vec<f32>)> {
    // Convert Python dict to Rust HashMap
    let mut rust_data_dict: HashMap<i32, Array2<f32>> = HashMap::new();
    
    let dict = data_dict.as_ref(py);
    for item in dict.items()? {
        let (key, value): (i32, &PyAny) = item.extract()?;
        let array = value.downcast::<PyArray2<f32>>()?;
        let ndarray = unsafe { array.as_array().to_owned() };
        rust_data_dict.insert(key, ndarray);
    }
    
    // Print the last numpy array
    if let Some((&last_key, last_array)) = rust_data_dict.iter().max_by_key(|(&k, _)| k) {
        println!("Last numpy array (key={}):", last_key);
        println!("{:?}", last_array);
    }
    
    // Call the original Rust function
    let result = multi_level_window(
        base_i,
        base_j,
        current_level,
        &rust_data_dict,
        nb_size,
        last_nb_size,
    );
    
    Ok(result)
}

/// A Python module implemented in Rust.
#[pymodule]
fn multires_connectivity(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, py)?)?;
    m.add_function(wrap_pyfunction!(py_multi_level_window, py)?)?;  // Add the new function
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn multires_connectivity(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, py)?)?;
    m.add_function(wrap_pyfunction!(py_multi_level_window, py)?)?;  // Add the new function
    Ok(())
}
