use anyhow::{Result, bail};
use pyo3::types::{PyAny, PyAnyMethods};
use pyo3::Bound;
use numpy::{PyArray2, PyArray3};
use std::collections::HashMap;
use ndarray::{Array3, Array2};
use crate::affine::Affine;


/// Convert a Python dict of rasterio transforms into a Rust HashMap.
pub fn to_transform_map(data_dict: &Bound<PyAny>) -> Result<HashMap<i32, Affine>> {
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

    Ok(rust_map)
}


/// Convert a Python 2D mask array to Array2<bool>
pub fn to_mask<'py>(data: &Bound<'py, PyAny>) -> Result<Array2<bool>> {
    if data.is_none() {
        bail!("Mask must be provided!");
    }
    
    let py_array = data.extract::<&PyArray2<bool>>()
        .map_err(|e| anyhow::anyhow!("Failed to extract mask array: {}", e))?;
    let array_owned = unsafe { py_array.as_array().to_owned() };
    Ok(array_owned)
}


/// Convert a Python 2D numpy array into an Option<Arr> that can be a Rust `Array2<f32>`.
pub fn to_array<'py>(data: &Bound<'py, PyAny>) -> Result<Option<Array2<f32>>> {
    if data.is_none() {
        Ok(None) // Python passed `None`
    } else {
        let py_array = data.extract::<&PyArray2<f32>>()?;
        let array_owned = unsafe { py_array.as_array().to_owned() };
        Ok(Some(array_owned))
    }
}


/// Convert a Python 3D numpy array into an Option<Array3<f32>> that can be used in Rust.
pub fn to_array_3d<'py>(data: &Bound<'py, PyAny>) -> Result<Option<Array3<f32>>> {
    if data.is_none() {
        Ok(None) // Python passed `None`
    } else {
        let py_array = data.extract::<&PyArray3<f32>>()?;
        let array_owned = unsafe { py_array.as_array().to_owned() };
        Ok(Some(array_owned))
    }
}
