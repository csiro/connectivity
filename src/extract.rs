use anyhow::{Result, bail};
use numpy::{PyReadonlyArray2, PyReadonlyArray3};
use pyo3::Bound;
use pyo3::types::{PyAny, PyAnyMethods, PyList, PyListMethods};
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


/// Convert a Python list of (exterior, holes) ring pairs into a Rust-native
/// `Vec<(exterior, holes)>` of (x, y) world-coordinate vertices.
///
/// Expected Python structure:
///   [(exterior_xy: ndarray (N, 2), [hole_xy: ndarray (M, 2), ...]), ...]
pub fn to_polygon_rings<'py>(
    data: &Bound<'py, PyAny>,
) -> Result<Vec<(Vec<(f64, f64)>, Vec<Vec<(f64, f64)>>)>> {
    let list = data
        .cast::<PyList>()
        .map_err(|e| anyhow::anyhow!("Expected a list of polygons: {}", e))?;

    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let (ext_obj, holes_obj) = item
            .extract::<(Bound<PyAny>, Bound<PyAny>)>()
            .map_err(|e| anyhow::anyhow!("Each polygon must be (exterior, holes): {}", e))?;

        let exterior = extract_ring(&ext_obj)?;

        let holes_list = holes_obj
            .cast::<PyList>()
            .map_err(|e| anyhow::anyhow!("Holes must be a list of arrays: {}", e))?;
        let mut holes = Vec::with_capacity(holes_list.len());
        for h in holes_list.iter() {
            holes.push(extract_ring(&h)?);
        }

        out.push((exterior, holes));
    }
    Ok(out)
}


/// Extract a ring of (x, y) vertices from a numpy (N, 2) f64 array.
fn extract_ring<'py>(data: &Bound<'py, PyAny>) -> Result<Vec<(f64, f64)>> {
    let arr = data
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|e| anyhow::anyhow!("Ring must be a 2D float64 array: {}", e))?;
    let view = arr.as_array();
    if view.shape()[1] != 2 {
        bail!("Ring array must have shape (N, 2), got {:?}", view.shape());
    }
    Ok(view.outer_iter().map(|row| (row[0], row[1])).collect())
}
