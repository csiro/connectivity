use anyhow::{Result, bail};
use numpy::{PyReadonlyArray2, PyReadonlyArray3};
use pyo3::Bound;
use pyo3::types::{PyAny, PyAnyMethods};
use std::collections::HashMap;
use ndarray::{Array3, Array2};
use crate::affine::Affine;


/// Convert a Python dict of rasterio transforms into a Rust HashMap.
pub fn to_transform_map(data_dict: &Bound<PyAny>) -> Result<HashMap<i32, Affine>> {
    let mut rust_map = HashMap::new();

    for pair in data_dict.call_method0("items")?.try_iter()? {
        let (key, transform_tuple) =
            pair?.extract::<(i32, (f64, f64, f64, f64, f64, f64))>()?;
        rust_map.insert(key, transform_tuple.into());
    }

    Ok(rust_map)
}


/// Convert a Python 2D mask array to Array2<bool>
pub fn to_mask<'py>(data: &Bound<'py, PyAny>) -> Result<Array2<bool>> {
    if data.is_none() {
        bail!("Mask must be provided!");
    }
    
    let py_array = data.extract::<PyReadonlyArray2<'_, bool>>()
        .map_err(|e| anyhow::anyhow!("Failed to extract mask array: {}", e))?;
    let array_owned = py_array.as_array().to_owned();
    Ok(array_owned)
}


/// Convert a Python 2D numpy array into an Option<Arr> that can be a Rust `Array2<f32>`.
pub fn to_array<'py>(data: &Bound<'py, PyAny>) -> Result<Option<Array2<f32>>> {
    if data.is_none() {
        Ok(None) // Python passed `None`
    } else {
        let py_array = data.extract::<PyReadonlyArray2<'_, f32>>()
            .map_err(|e| anyhow::anyhow!("Failed to extract 2D array: {}", e))?;
        let array_owned = py_array.as_array().to_owned();
        Ok(Some(array_owned))
    }
}


/// Convert a Python 3D numpy array into an Option<Array3<f32>> that can be used in Rust.
pub fn to_array_3d<'py>(data: &Bound<'py, PyAny>) -> Result<Option<Array3<f32>>> {
    if data.is_none() {
        Ok(None) // Python passed `None`
    } else {
        let py_array = data.extract::<PyReadonlyArray3<'_, f32>>()
            .map_err(|e| anyhow::anyhow!("Failed to extract 3D array: {}", e))?;
        let array_owned = py_array.as_array().to_owned();
        Ok(Some(array_owned))
    }
}
