use pyo3::types::{PyAny, PyAnyMethods};
use pyo3::Bound;
use pyo3::PyResult;
use numpy::{PyArray2, PyArray3};
use ndarray::{Array3, Array2, Array1, s};
use std::collections::HashMap;
// local module
use crate::affine::Affine;


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
        if i < array3.shape()[0] && j < array3.shape()[1] {
            return array3.slice(s![i, j, ..]).to_owned(); // returns Array1<f32>
        }
    }
    // Return empty array if anything fails
    Array1::zeros(0)
}

