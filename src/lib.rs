use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3::exceptions::PyRuntimeError;
use pyo3::Bound;
use std::collections::HashMap;
use numpy::{PyArray2, ToPyArray};
use ndarray::{Array3, Array2};
// local modules
mod utils;
mod extract;
mod window;
mod builder;
mod metrics;
mod distances;
mod affine;
mod graph;
mod routing;
mod core;
use affine::Affine;


/// Compute habitat/PARC connectivity and BERI.
///
/// # Arguments
/// * `condition` - 2D array of habitat-condition values in [0, 1]. Values may be
///   pre-scaled; higher values represent better condition.
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
/// * `window_size` - Radius of the local neighborhood (in pixels) at the finest
///   resolution. Must be an odd number (e.g. 3 for a 3×3 window).
/// * `outer_window` - Radius of the neighborhood at the coarsest resolution level,
///   capturing broader connectivity context. Must be odd and ≥ `window_size`.
/// * `n_threads` - Optional number of CPU threads to use. If `None`, all available
///   cores are used.
///
/// # Returns
/// A 2D array of connectivity values for each cell at the native resolution.
#[pyfunction(signature = (condition, pa_array, transgrid_list, transforms, lambdas, is_geo, max_cost, window_size, outer_window, n_threads=None))]
fn connectivity(
    condition: &Bound<PyAny>,
    pa_array: &Bound<PyAny>,
    transgrid_list: &Bound<PyAny>,
    transforms: &Bound<PyAny>,
    lambdas: Vec<f32>,
    is_geo: bool,
    max_cost: f32,
    window_size: i32,
    outer_window: i32,
    n_threads: Option<usize>,
) -> PyResult<Py<PyArray2<f32>>> {

    // Create a Rust HashMap from Py data
    let cond_map: HashMap<i32, Array2<f32>> = extract::to_2d_map(condition);

    // Convert Python list into a native Rust Vec<HashMap<i32, Array3<f32>>>
    let list = transgrid_list.downcast::<PyList>()?; // Convert PyAny to PyList
    let trans_maps: Vec<HashMap<i32, Array3<f32>>> = list.iter().map(|item| {
        let dict = item.downcast::<PyDict>().unwrap(); // handling errors properly
        extract::to_3d_map(dict)
    }).collect();

    // Convert the tranform of each level for coordinate and distance calcaulations
    let transform_map: HashMap<i32, Affine> = extract::to_transform_map(transforms);

    let override_array = extract::to_array(pa_array)?;

    // Set the number of cores for parallel processing with Rayon
    // Use a local pool as a safe way in PyO3 contexts
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads.unwrap_or(num_cpus::get()).max(1))
        .build()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

 
    // Core connectivity function
    let outarray: Array2<f32> = pool
        .install(|| core::conn(
            &cond_map,
            &trans_maps,
            &transform_map,
            &override_array,
            &lambdas,
            is_geo,
            max_cost,
            window_size,
            outer_window,
        ))
        .map_err(|er| PyRuntimeError::new_err(format!("Climsim failed: {er}")))?;

    // Convert the array back to Python with gil
    Python::with_gil(|py| {
        let pyarray = outarray.to_pyarray_bound(py);
        Ok(pyarray.unbind())
    })
}


#[pymodule]
fn rust_conn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction_bound!(connectivity, m)?)?;
    Ok(())
}

