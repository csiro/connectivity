use ndarray::{Array2, Array3};
use numpy::{PyArray2, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use pyo3::Bound;
use std::collections::HashMap;
// local modules
mod affine;
mod builder;
mod core;
mod distances;
mod extract;
mod graph;
mod inpaint;
mod metrics;
mod overview;
mod routing;
mod utils;
mod window;
use affine::Affine;

/// Compute habitat/PARC connectivity and BERI.
///
/// # Arguments
/// * `condition` - 2D array of habitat-condition values in [0, 1]. Values are
///   pre-scaled (Python-side); higher values represent better condition.
/// * `pa_array` - Optional 2D array of protected-area (PA) proportions. If not `None`,
///   PARC-connectedness is computed; if `None`, only habitat connectedness is returned.
/// * `transgrid_list` - List of transition / cost grids (one per resolution level) used
///   for multi-scale connectivity calculations.
/// * `transforms` - List of spatial transforms corresponding to `transgrid_list`.
/// * `lambdas` - Bandwidth values for the connectivity kernels, controlling the
///   effective dispersal distance (e.g. [2.0, 20.0, 200.0]).
/// * `is_geo` - Whether coordinates are geographic (lat/lon) rather than projected.
/// * `max_cost` - Relative cost of moving through completely degraded cells
///   (condition = 0). Applied as `w = (1.0 - max_cost) * condition + max_cost`.
/// * `window_size` - Odd local window width used for non-coarsest levels
///   (e.g. 3 gives an effective 6×6 current-level window).
/// * `outer_window` - Odd coarsest-level window width used to set the long-range
///   search reach. Must be ≥ `window_size`.
/// * `offsets` -
/// * `window_mode` - "block" for the original snapped multi-resolution windows,
///   "fractional" for source-centered square annuli, or "circular" for
///   source-centered circular annuli.
/// * `n_threads` - Optional number of CPU threads to use. If `None`, all available
///   cores are used.
///
/// # Returns
/// A 2D array of connectivity values for each cell at the native resolution.
#[pyfunction(signature = (condition, mask, transgrid_list, transforms, levels, lambdas, is_geo, max_cost, window_size, outer_window, offsets, n_threads=None, window_mode="block"))]
fn connectivity(
    condition: &Bound<PyAny>,
    mask: &Bound<PyAny>,
    transgrid_list: &Bound<PyAny>,
    transforms: &Bound<PyAny>,
    levels: Vec<usize>,
    lambdas: Vec<f32>,
    is_geo: bool,
    max_cost: f32,
    window_size: i32,
    outer_window: i32,
    offsets: (usize, usize),
    n_threads: Option<usize>,
    window_mode: &str,
) -> PyResult<Py<PyArray2<f32>>> {
    // Get the Numpy array
    let cond_array = extract::to_array(condition)
        .map_err(|er| PyRuntimeError::new_err(format!("Reading condition failed: {er}")))?;

    // Convert Python list into a native Rust Vec<Array3<f32>> or None
    let trans_arrays: Option<Vec<Array3<f32>>> = if transgrid_list.is_none() {
        None
    } else {
        let arrays: Vec<Array3<f32>> = transgrid_list
            .downcast::<PyList>()?
            .iter()
            .map(|item| extract::to_array_3d(&item).unwrap())
            .filter_map(|opt| opt)
            .collect();

        (!arrays.is_empty()).then_some(arrays)
    };

    // Convert the tranform of each level for coordinate and distance calcaulations
    let transform_map: HashMap<i32, Affine> =
        extract::to_transform_map(transforms).map_err(|er| {
            PyRuntimeError::new_err(format!("Faild getting transform information: {er}"))
        })?;

    let mask_array = extract::to_mask(mask)
        .map_err(|er| PyRuntimeError::new_err(format!("Reading mask failed: {er}")))?;

    let window_mode = window::WindowMode::parse(window_mode).map_err(PyValueError::new_err)?;

    // Set the number of cores for parallel processing with Rayon
    // Use a local pool to safely control exact number of core
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads.unwrap_or(num_cpus::get()).max(1))
        .build()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Core connectivity function
    let outarray: Array2<f32> = pool
        .install(|| {
            core::conn(
                &cond_array,
                trans_arrays.as_deref(),
                &transform_map,
                &mask_array,
                &levels,
                &lambdas,
                is_geo,
                max_cost,
                window_size,
                outer_window,
                offsets,
                window_mode,
            )
        })
        .map_err(|er| PyRuntimeError::new_err(format!("Connectivity failed: {er}")))?;

    // Convert the array back to Python with gil
    Python::with_gil(|py| {
        let pyarray = outarray.to_pyarray_bound(py);
        Ok(pyarray.unbind())
    })
}

#[pyfunction(signature = (img, size=11, max_iter=200, tol=1e-3, init="nearest", n_threads=None))]
fn inpaint_nans_diffusion(
    img: &Bound<PyAny>,
    size: usize,
    max_iter: usize,
    tol: f32,
    init: &str,
    n_threads: Option<usize>,
) -> PyResult<Py<PyArray2<f32>>> {
    if init != "nearest" && init != "mean" {
        return Err(PyValueError::new_err("init must be 'nearest' or 'mean'"));
    }

    let img_array = extract::to_array(img)
        .map_err(|er| PyRuntimeError::new_err(format!("Reading image failed: {er}")))?
        .ok_or_else(|| PyValueError::new_err("img must be a 2D numpy array"))?;

    // Use a local Rayon pool to respect caller-provided thread count.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads.unwrap_or(num_cpus::get()).max(1))
        .build()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let outarray: Array2<f32> = pool
        .install(|| inpaint::inpaint_nans_diffusion(&img_array, size, max_iter, tol, init))
        .map_err(|er| {
            if er.to_string().contains("All pixels are NaN") {
                PyValueError::new_err(er.to_string())
            } else {
                PyRuntimeError::new_err(format!("Inpainting failed: {er}"))
            }
        })?;

    Python::with_gil(|py| {
        let pyarray = outarray.to_pyarray_bound(py);
        Ok(pyarray.unbind())
    })
}

#[pymodule]
fn rust_conn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction_bound!(connectivity, m)?)?;
    m.add_function(wrap_pyfunction_bound!(inpaint_nans_diffusion, m)?)?;
    Ok(())
}
