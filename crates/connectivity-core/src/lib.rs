//! Efficient multi-resolution least-cost algorithm for connectivity-based biodiversity
//! indicators on in-memory raster arrays.
//!
//! This is the numerical engine shared by the Python package
//! [`eco-connectivity`](https://pypi.org/project/eco-connectivity/) and the R package
//! [`connectivity`](https://github.com/csiro/connectivity/tree/main/r/connectivity). For every
//! unmasked focal cell of a raster tile, the [`connectivity`] function computes
//! - habitat or PARC **connectedness**, when no transgrids are given, or
//! - the Bioclimatic Ecosystem Resilience Index (**BERI**), when transgrids are given,
//!
//! and [`pixel_coverage`] computes the fraction of each pixel covered by polygons, e.g. to
//! build protected-area proportion rasters. The crate has no Python, R, GDAL or file I/O
//! dependencies: callers read rasters and pass plain [`ndarray`] arrays.
//!
//! # Method
//! Each level `L` of [`Config::levels`] aggregates `L x L` base pixels into overview cells.
//! Around a focal pixel, every level contributes the overview cells of an annulus (see
//! [`Config::window_size`]), so detail is kept near the cell and distant areas are coarse.
//! These cells form a graph with queen-neighbour edges within a level and links from the rim
//! of each level to the next coarser one. Two integer Dijkstra searches from the focal cell
//! give, for every reached cell `v`, the condition-adjusted path length `a_v` and the intact
//! path length `g_v` in kilometres: an edge into `v` of length `d` costs `w_v * d` and `d`,
//! with `w_v = (1 - max_cost) * t_v + max_cost` for traversal value `t_v`, rounded to integer
//! units of 10 m for routing. With `q_v` the condition, `A_v` the habitat area weight and
//! `p_v` the protected fraction (1 unless [`Config::pa_to_pa`]), connectedness for one lambda
//! is
//!
//! `sum_v exp(-a_v / lambda) q_v p_v A_v / sum_v exp(-g_v / lambda) A_v`.
//!
//! BERI uses Gaussian kernels `exp(-(x / lambda)^2 / 283.465)` instead, without `p_v`, and
//! weights each scenario's numerator by that scenario's climate similarity and the
//! denominator by the current climate's. A similarity is `exp(-L1 distance)` between the
//! focal cell's current-climate bands and the cell's scenario bands, and scenario
//! numerators are aggregated as `0.5 * (mean + min)` over all scenarios, current climate
//! included. The output is the mean over [`Config::lambdas`].
//!
//! # Inputs
//! All 2-D arrays are `(rows, cols)` grids of one tile with the same shape, indexed
//! `[[row, col]]` from the tile's top-left pixel; NaN marks missing values. Transgrids add a
//! trailing band axis. See [`Inputs`] for each array.
//!
//! **Coordinates and units.** [`Affine`] maps (row, col) pixel centres to world coordinates.
//! Projected grids (`is_geo = false`) must be in metres; geographic grids (`is_geo = true`)
//! are longitude/latitude in degrees, with great-circle distances on a 6371 km sphere and
//! habitat area weighted by cos(latitude). Distances and lambdas are in kilometres.
//! Transform tuples follow the rasterio/`affine` order `(a, b, c, d, e, f)`, i.e.
//! `x = a * col + b * row + c` and `y = d * col + e * row + f`, not GDAL's geotransform order.
//!
//! **Tiles and offsets.** [`Config::offsets`] is the global `(row0, col0)` of the tile's
//! top-left pixel in the full raster. Overview cells are anchored to the full raster's grid:
//! level-`L` cell `(R, C)` covers global base rows `R * L .. (R + 1) * L` and columns
//! `C * L .. (C + 1) * L`, and a tile's level-`L` grid covers every such cell that intersects
//! the tile, averaging only the tile's pixels in cells cut by its edge. The level-`L`
//! transform must map that grid ([`level_transforms`] builds it from the full raster's base
//! transform), so tiles evaluated separately with enough context around their cores join
//! without seams.
//!
//! # Caller responsibilities
//! The core aggregates overviews from base-resolution arrays: averages of condition,
//! traversal and transgrids, habitat area weights (valid-cell counts, or cos-latitude areas)
//! and protected fractions. Callers, such as the Python package, remain responsible for:
//! - reading rasters, detecting geographic CRSs, converting nodata to NaN and scaling
//!   condition, protected-area (PA) and resistance values to [0, 1];
//! - planning tiles, their context padding and read windows, and passing matching offsets
//!   and level transforms;
//! - building the mask from polygons, nodata and, for PARC, non-protected cells;
//! - PARC preparation: `condition = max(condition, pa)` and
//!   `mask = !(pa > 0 && condition is finite)`;
//! - resistance surfaces: `traversal = clip(1 - resistance, 0, 1)`, falling back to condition
//!   where resistance is missing and NaN where condition is missing;
//! - BERI inputs: current-climate transgrid first, then future scenarios, bands last;
//! - normalising levels (add 1, round to powers of two, require a continuous sequence) and
//!   `outer_window >= window_size`, which the Python wrappers check before calling;
//! - post-processing: Connected Habitat for habitat (not PARC) connectedness with Python's
//!   `option` 1 (connectedness), 2 (condition x connectedness) or 3 (the default,
//!   sqrt(condition x connectedness)); cropping tiles back to their polygons; writing rasters.
//!
//! # Threads and memory
//! Work runs on the current [Rayon](https://docs.rs/rayon) thread pool: the global pool, or
//! the pool of an enclosing `ThreadPool::install`, which is how the Python binding applies
//! `n_threads`. Outputs are bit-identical for any thread count on the same build and
//! platform. Besides the overviews and the output, a call keeps a shared cache of window
//! templates and distance tables of at most 128 MiB and one scratch buffer per pool thread;
//! focal cells reuse those buffers.
//!
//! # Example
//! ```
//! use connectivity_core::ndarray::Array2;
//! use connectivity_core::{connectivity, level_transforms, Config, Inputs};
//!
//! // A 24 x 24 tile at the top-left of a projected raster with 100 m pixels
//! let condition = Array2::from_shape_fn((24, 24), |(i, j)| ((i * 7 + j * 3) % 10) as f32 / 10.0);
//! let mask = condition.mapv(f32::is_nan);
//! let levels = [1, 2, 4];
//! let base = (100.0, 0.0, 500_000.0, 0.0, -100.0, 6_000_000.0);
//! let transforms = level_transforms(base, &levels, (0, 0));
//! let inputs = Inputs {
//!     condition: &condition,
//!     mask: &mask,
//!     traversal: None,
//!     transgrids: None,
//!     transforms: &transforms,
//! };
//! let config = Config { levels: &levels, outer_window: 5, ..Config::default() };
//! let out = connectivity(&inputs, &config).unwrap();
//! assert_eq!(out.dim(), (24, 24));
//! assert!(out.iter().all(|v| (0.0..=1.0).contains(v)));
//! ```
mod affine;
mod builder;
mod core;
mod coverage;
mod distances;
mod graph;
#[cfg(test)]
mod legacy;
mod metrics;
mod overview;
mod routing;
mod tables;
mod template;
#[cfg(test)]
mod testkit;
mod utils;
#[cfg(test)]
mod verify;
mod window;

pub use affine::{level_transforms, Affine};
pub use coverage::{pixel_coverage, Polygon};
/// The `ndarray` version of this crate's array types.
pub use ndarray;
pub use window::WindowMode;

use ndarray::{Array2, Array3};
use std::collections::HashMap;

/// Borrowed arrays of one tile for [`connectivity`]. 2-D arrays share the `(rows, cols)`
/// shape of `condition`.
#[derive(Clone, Copy, Debug)]
pub struct Inputs<'a> {
    /// Habitat condition, expected in [0, 1] with higher values for better condition. It sets
    /// each destination's value and habitat area, and the path weights unless `traversal` is
    /// given. NaN cells are left out of habitat area and of the graph; coarser overview cells
    /// average the finite cells they contain.
    pub condition: &'a Array2<f32>,
    /// `true` skips a focal cell, whose output stays NaN. Masked cells with finite condition
    /// still serve as destinations and path cells. Under [`Config::pa_to_pa`], `false` also
    /// marks protected cells. Unmasked cells with NaN condition reach nothing and score 0, so
    /// callers normally mask them.
    pub mask: &'a Array2<bool>,
    /// Optional traversal values in [0, 1], high for easy passage (e.g. `1 - resistance`),
    /// used only for path weights. `None` uses condition; cells whose traversal overview is
    /// NaN fall back to condition.
    pub traversal: Option<&'a Array2<f32>>,
    /// BERI scenarios: the current climate first, then future scenarios. Each array is
    /// `(rows, cols, bands)` with condition's rows and columns and the same bands in the same
    /// order; any memory layout works, e.g. a band-major view. A cell whose bands are not all
    /// finite has similarity 0. `None` or an empty slice computes connectedness instead.
    pub transgrids: Option<&'a [Array3<f32>]>,
    /// Transform of each level's grid, keyed by level; see [`level_transforms`].
    pub transforms: &'a HashMap<i32, Affine>,
}

/// Settings of one [`connectivity`] call. `Config::default()` gives the Python defaults,
/// with level 1 included as the core requires.
#[derive(Clone, Copy, Debug)]
pub struct Config<'a> {
    /// Overview levels in base pixels per overview cell: powers of two forming a continuous
    /// sequence that includes 1, such as `[1, 2, 4, 8, 16, 32]`, in any order.
    pub levels: &'a [usize],
    /// Dispersal kernel bandwidths in kilometres; the output is the mean of their scores. An
    /// empty slice scores every evaluated cell 0.
    pub lambdas: &'a [f32],
    /// Whether transforms are geographic (degrees) rather than projected (metres).
    pub is_geo: bool,
    /// Relative cost of crossing a fully degraded cell: a cell with traversal value `t` has
    /// path weight `(1 - max_cost) * t + max_cost`, so `t = 1` costs 1 and `t = 0` costs
    /// `max_cost`.
    pub max_cost: f32,
    /// Odd window size of every level but the coarsest. The level-`L` annulus spans
    /// `L * window_size / 2` to `L * window_size` base pixels from the focal pixel centre, and
    /// level 1 is the disc of radius `window_size`; in square mode these are half-widths.
    pub window_size: i32,
    /// Odd window size of the coarsest level `L`, whose annulus reaches `L * outer_window`
    /// base pixels. Values below `window_size` use `window_size`.
    pub outer_window: i32,
    /// Global `(row0, col0)` of the tile's top-left pixel in the full raster, in base pixels;
    /// `(0, 0)` for a whole raster.
    pub offsets: (usize, usize),
    /// Circular or square annuli.
    pub window_mode: WindowMode,
    /// PARC target gating: only protected destinations, weighted by each overview cell's
    /// protected fraction (unmasked share of its valid pixels), count in the connectedness
    /// numerator. Paths and the denominator are unchanged, and a focal cell that reaches no
    /// protected destination scores 0. For BERI it only makes focal cells without neighbours
    /// score 0; Python's `beri()` never sets it.
    pub pa_to_pa: bool,
}

impl Default for Config<'_> {
    fn default() -> Self {
        Self {
            levels: &[1, 2, 4, 8, 16, 32],
            lambdas: &[2.0, 20.0, 200.0],
            is_geo: false,
            max_cost: 2.0,
            window_size: 3,
            outer_window: 9,
            offsets: (0, 0),
            window_mode: WindowMode::Circular,
            pa_to_pa: false,
        }
    }
}

/// Connectedness, or BERI when [`Inputs::transgrids`] is non-empty, of every unmasked cell.
///
/// Returns an array with the shape of [`Inputs::condition`]: NaN where the mask is `true`,
/// otherwise the mean score over [`Config::lambdas`], normally in [0, 1] and 0 for a cell that
/// reaches no destination.
///
/// # Errors
/// Returns an error when [`Inputs::traversal`] differs in shape from the condition array
/// (`"Dimension mismatch for key <level>: <traversal shape> vs <condition shape>"`). The
/// Python binding raises errors as `RuntimeError("Connectivity failed: <message>")`.
///
/// # Panics
/// Invalid settings panic, with messages the Python binding raises as
/// `pyo3_runtime.PanicException`; panics on worker threads reach the caller through Rayon.
/// - A level that is not a power of two (`"Failed average resampling.: Level 3 is not a power
///   of 2"`), or no level-1 transform (`"Missing base-level transform (key=1)."`).
/// - When at least one cell is unmasked: no level 1 (`"Base condition level not found!"`),
///   an even `window_size` (`"Window size must be odd-numbered"`) or an even
///   `outer_window >= window_size` (`"Outer window size must be odd-numbered"`), a level
///   without a transform (`"Missing level in Affine set."`), or levels that skip a power of
///   two once a level past the gap has window cells (`"levels must form a continuous
///   power-of-two sequence"`).
/// - An integer path cost beyond `u32::MAX`, e.g. from a very large `max_cost` or unscaled
///   traversal values (`"Path cost overflow; check max_cost and resistance scaling!"`).
/// - A mask or transgrid smaller than the condition array can panic on indexing.
pub fn connectivity(inputs: &Inputs, config: &Config) -> anyhow::Result<Array2<f32>> {
    core::conn(inputs, config)
}
