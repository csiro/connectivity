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
mod coverage;
mod distances;
mod extract;
mod graph;
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
/// * `mask` - 2D boolean array. `true` marks cells excluded from analysis (skipped as
///   focal sources); in the PARC edition `false` marks protected-area cells.
/// * `transgrid_list` - List of transition / cost grids (one per resolution level) used
///   for multi-scale connectivity calculations.
/// * `transforms` - List of spatial transforms corresponding to `transgrid_list`.
/// * `lambdas` - Bandwidth values for the connectivity kernels, controlling the
///   effective dispersal distance (e.g. [2.0, 20.0, 200.0]).
/// * `is_geo` - Whether coordinates are geographic (lat/lon) rather than projected.
/// * `max_cost` - Relative cost of moving through a fully-degraded cell (traversal value 0).
///   The path weight is `w = (1.0 - max_cost) * t + max_cost`, where `t` is the traversal
///   value: the `condition` value by default, or the (pre-inverted) `traversal` raster when
///   supplied. `t = 1` gives `w = 1` (cheapest), `t = 0` gives `w = max_cost` (costliest).
/// * `window_size` - Odd local window width used for non-coarsest levels
///   (e.g. 3 gives an effective 6×6 current-level window).
/// * `outer_window` - Odd coarsest-level window width used to set the long-range
///   search reach. Must be ≥ `window_size`.
/// * `offsets` -
/// * `window_mode` - "circular" for source-centered circular annuli or
///   "square" for source-centered square annuli.
/// * `traversal` - Optional 2D array (same shape as `condition`) of pre-inverted, [0, 1]
///   traversal values (high = easy passage, i.e. `1 - resistance`) used only for the path
///   weight `w`. `None` falls back to `condition`, giving bit-identical output. Indicator
///   values (habitat area and the condition multiplier) always use `condition`, not this.
/// * `pa_to_pa` - PARC edition only. When `true`, only nodes on a protected area are
///   counted as destinations (weighted by their protected fraction); paths still route through
///   the non-PA landscape. An isolated PA scores 0. When `false`, all reachable nodes count.
/// * `n_threads` - Optional number of CPU threads to use. If `None`, all available
///   cores are used.
///
/// # Returns
/// A 2D array of connectivity values for each cell at the native resolution.
#[pyfunction(signature = (condition, mask, transgrid_list, transforms, levels, lambdas, is_geo, max_cost, window_size, outer_window, offsets, traversal=None, n_threads=None, window_mode="circular", pa_to_pa=false))]
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
    traversal: Option<Bound<PyAny>>,
    n_threads: Option<usize>,
    window_mode: &str,
    pa_to_pa: bool,
) -> PyResult<Py<PyArray2<f32>>> {
    // Get the Numpy array
    let cond_array = extract::to_array(condition)
        .map_err(|er| PyRuntimeError::new_err(format!("Reading condition failed: {er}")))?;

    // Convert Python list into a native Rust Vec<Array3<f32>> or None
    let trans_arrays: Option<Vec<Array3<f32>>> = if transgrid_list.is_none() {
        None
    } else {
        let arrays: Vec<Array3<f32>> = transgrid_list
            .cast::<PyList>()?
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

    // Optional traversal raster (resistance-derived; pre-inverted Python-side). `None` when
    // absent, in which case the graph weight falls back to condition.
    let trav_array: Option<Array2<f32>> = match traversal {
        Some(obj) => extract::to_array(&obj)
            .map_err(|er| PyRuntimeError::new_err(format!("Reading traversal failed: {er}")))?,
        None => None,
    };

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
                trav_array.as_ref(),
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
                pa_to_pa,
            )
        })
        .map_err(|er| PyRuntimeError::new_err(format!("Connectivity failed: {er}")))?;

    // Convert the array back to Python with gil
    Python::attach(|py| {
        let pyarray = outarray.to_pyarray(py);
        Ok(pyarray.unbind())
    })
}


/// Compute the proportion of each raster pixel covered by a (multi)polygon.
///
/// # Arguments
/// * `polygons` - Python list of `(exterior_xy, [hole_xy, ...])` tuples, where each
///   ring is an `(N, 2)` float64 numpy array of (x, y) world-coordinate vertices.
///   Polygons are assumed already-unioned on the Python side.
/// * `transform` - Affine 6-tuple `(a, b, c, d, e, f)` of the reference raster.
/// * `shape` - Output raster shape `(nrows, ncols)`.
/// * `n_threads` - Optional number of CPU threads to use. If `None`, all available
///   cores are used.
///
/// # Returns
/// A 2D float32 array of shape `shape` with values in [0, 1].
#[pyfunction(signature = (polygons, transform, shape, n_threads=None))]
fn pixel_coverage(
    polygons: &Bound<PyAny>,
    transform: (f64, f64, f64, f64, f64, f64),
    shape: (usize, usize),
    n_threads: Option<usize>,
) -> PyResult<Py<PyArray2<f32>>> {
    let rings = extract::to_polygon_rings(polygons)
        .map_err(|er| PyRuntimeError::new_err(format!("Reading polygons failed: {er}")))?;
    let tr: Affine = transform.into();
    let (nrows, ncols) = shape;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads.unwrap_or(num_cpus::get()).max(1))
        .build()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let outarray = pool.install(|| coverage::pixel_coverage_array(&rings, &tr, nrows, ncols));

    Python::attach(|py| {
        let pyarray = outarray.to_pyarray(py);
        Ok(pyarray.unbind())
    })
}

#[pymodule(gil_used = true)]
fn rust_conn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(connectivity, m)?)?;
    m.add_function(wrap_pyfunction!(pixel_coverage, m)?)?;
    Ok(())
}
