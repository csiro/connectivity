use connectivity_core::ndarray::{Array2, Array3};
use connectivity_core::{level_transforms, Config, Inputs, WindowMode};
use extendr_api::prelude::*;
use extendr_api::Result;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn error(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

// R vectors are column-major. Copy into owned arrays before entering Rayon:
// worker threads must never access R objects or call the R API.
fn matrix(values: &[f64], rows: usize, cols: usize) -> Result<Array2<f32>> {
    if values.len() != rows * cols {
        return Err(error("Matrix dimensions do not match its data"));
    }
    Ok(Array2::from_shape_fn((rows, cols), |(row, col)| {
        let value = values[row + rows * col];
        if value.is_nan() {
            f32::NAN
        } else {
            value as f32
        }
    }))
}

/// Internal native interface; use connectivity_matrix() from R.
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn connectivity_matrix_impl(
    condition: Vec<f64>,
    mask: Vec<i32>,
    shape: Vec<i32>,
    transform: Vec<f64>,
    traversal: Vec<f64>,
    transgrids: List,
    bands: i32,
    levels: Vec<i32>,
    lambdas: Vec<f64>,
    is_geo: bool,
    max_cost: f64,
    window_size: i32,
    outer_window: i32,
    offsets: Vec<i32>,
    window_mode: &str,
    pa_to_pa: bool,
    n_threads: i32,
) -> Result<Vec<f64>> {
    if shape.len() != 2
        || shape.iter().any(|&n| n <= 0)
        || transform.len() != 6
        || offsets.len() != 2
        || offsets.iter().any(|&n| n < 0)
        || n_threads < 0
    {
        return Err(error(
            "Invalid dimensions, transform, offsets or thread count",
        ));
    }
    let (rows, cols) = (shape[0] as usize, shape[1] as usize);
    let condition = matrix(&condition, rows, cols)?;
    if mask.len() != rows * cols {
        return Err(error("Mask dimensions do not match condition"));
    }
    let mask = Array2::from_shape_fn((rows, cols), |(row, col)| mask[row + rows * col] != 0);
    let traversal = if traversal.is_empty() {
        None
    } else {
        Some(matrix(&traversal, rows, cols)?)
    };
    let mut grids = Vec::with_capacity(transgrids.len());
    for (_, grid) in transgrids.iter() {
        let values = grid
            .as_real_slice()
            .ok_or_else(|| error("Transgrids must be numeric arrays"))?;
        if bands <= 0 || values.len() != rows * cols * bands as usize {
            return Err(error("Transgrid dimensions do not match condition"));
        }
        grids.push(Array3::from_shape_fn(
            (rows, cols, bands as usize),
            |(row, col, band)| {
                let value = values[row + rows * col + rows * cols * band];
                if value.is_nan() {
                    f32::NAN
                } else {
                    value as f32
                }
            },
        ));
    }
    // Everything used by worker threads below is owned Rust data.
    let levels: Vec<usize> = levels
        .into_iter()
        .map(|level| {
            usize::try_from(level)
                .ok()
                .filter(|n| n.is_power_of_two())
                .ok_or_else(|| error("Levels must be positive powers of two"))
        })
        .collect::<Result<_>>()?;
    let lambdas: Vec<f32> = lambdas.into_iter().map(|value| value as f32).collect();
    if lambdas.is_empty()
        || lambdas.iter().any(|&v| !v.is_finite() || v <= 0.0)
        || !(max_cost as f32).is_finite()
    {
        return Err(error(
            "Lambdas and max_cost must be representable as finite float32 values",
        ));
    }
    let offsets = (offsets[0] as usize, offsets[1] as usize);
    let base = (
        transform[0],
        transform[1],
        transform[2],
        transform[3],
        transform[4],
        transform[5],
    );
    let transforms = level_transforms(base, &levels, offsets);
    let inputs = Inputs {
        condition: &condition,
        mask: &mask,
        traversal: traversal.as_ref(),
        transgrids: if grids.is_empty() { None } else { Some(&grids) },
        transforms: &transforms,
    };
    let config = Config {
        levels: &levels,
        lambdas: &lambdas,
        is_geo,
        max_cost: max_cost as f32,
        window_size,
        outer_window,
        offsets,
        window_mode: WindowMode::parse(window_mode).map_err(error)?,
        pa_to_pa,
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads as usize)
        .build()
        .map_err(|e| error(e.to_string()))?;
    let output = catch_unwind(AssertUnwindSafe(|| {
        pool.install(|| connectivity_core::connectivity(&inputs, &config))
    }))
    .map_err(|panic| {
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("Rust core panicked");
        error(format!("Connectivity failed: {message}"))
    })?
    .map_err(|e| error(format!("Connectivity failed: {e}")))?;

    // Return column-major data; the R wrapper restores dimensions and dimnames.
    let mut result = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            let value = output[[row, col]];
            result.push(if value.is_nan() {
                f64::na()
            } else {
                value as f64
            });
        }
    }
    Ok(result)
}

extendr_module! {
    mod ecoconnectivity;
    fn connectivity_matrix_impl;
}
