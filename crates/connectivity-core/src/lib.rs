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
//! # NoData
//! NaN condition cells (NoData, e.g. sea, lakes or rivers in global condition data) never carry
//! habitat area. [`Config::crossing`] sets how the graph treats them:
//! - `None` (the default, as in earlier versions): NoData is left out at level 1, but an
//!   overview cell is valid when any of its base cells is finite and coarse cells are linked by
//!   position alone, so coarse levels link land on both sides of a gap narrower than about twice
//!   their size, at no cost. Whether a focal cell uses such a link depends on its window, so a
//!   compact island near other land can have a linked part and a sealed part.
//! - [`CrossingMode::Landmass`]: separate landmasses are linked only across gaps at most
//!   [`Crossing::width`] kilometres wide; within one landmass the graph is as with `None`.
//! - [`CrossingMode::Strict`]: no gap wider than the width is crossed anywhere, including rivers
//!   and inlets inside one landmass; routes around a barrier within the window are kept.
//!
//! In both modes NoData in permitted gaps is crossed at the cost of [`Crossing::traversal`] and
//! adds no habitat area, protected fraction or climate similarity, and outputs without NoData are
//! bit-identical to `None`. See [`Crossing`] for how gaps are measured and how tiles behave.
//! The modes cost extra setup and memory: group labels for the cells of every level (about
//! 5.3 bytes per base cell) in landmass mode, two bytes per coarse cell plus a table of cells
//! holding several pieces in strict mode, and with permitted gaps the traversal overview. Strict mode also builds
//! graphs more slowly: on a 450 m test raster with levels 1 to 32, about 1.2 to 1.7 times as long
//! as without a mode, most at a width of 0, where coasts split coarse cells into the most pieces;
//! landmass mode takes about as long as without a mode.
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
//! - normalising levels (add 1, round to powers of two) and requiring
//!   `outer_window >= window_size`, as the Python wrappers do before calling: the core rejects
//!   levels that do not form a continuous sequence with an error, but uses `window_size` for a
//!   smaller `outer_window`;
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
mod crossing;
mod distances;
mod gate;
mod graph;
#[cfg(test)]
mod legacy;
mod metrics;
mod overview;
mod routing;
mod strict;
mod tables;
mod template;
#[cfg(test)]
mod testkit;
mod utils;
#[cfg(test)]
mod verify;
#[cfg(test)]
mod island;
mod window;

pub use affine::{level_transforms, Affine};
pub use coverage::{pixel_coverage, Polygon};
/// The `ndarray` version of this crate's array types.
pub use ndarray;
pub use window::WindowMode;

/// This crate's version, such as `"0.2.1"`, for callers that record which engine produced a
/// result, as the run logs of the Python and R packages do.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

use ndarray::{Array2, Array3};
use std::collections::HashMap;

/// Borrowed arrays of one tile for [`connectivity`]. 2-D arrays share the `(rows, cols)`
/// shape of `condition`.
#[derive(Clone, Copy, Debug)]
pub struct Inputs<'a> {
    /// Habitat condition, expected in [0, 1] with higher values for better condition. It sets
    /// each destination's value and habitat area, and the path weights unless `traversal` is
    /// given. NaN cells (NoData) carry no habitat area and coarser overview cells average the
    /// finite cells they contain; how the graph treats NoData is set by [`Config::crossing`]
    /// (see the crate docs, NoData).
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

/// How NoData cells may be crossed; see [`Crossing`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossingMode {
    /// Landmasses: land and crossable NoData that touch form groups, and a focal cell reaches
    /// only cells of its own group. Groups farther apart than the width are never linked;
    /// within a group the graph is as without a crossing mode, so coarse levels still link
    /// land across wider water inside one landmass (an inlet, or a river whose banks join).
    Landmass,
    /// Strict gap width: every overview cell is split into pieces, the 8-connected groups of
    /// its land and crossable cells, and pieces are linked only where their base cells touch
    /// (or, diagonally, both touch one piece of a cell sharing their corner). Every path then
    /// follows a base-resolution route that crosses no gap wider than the width, inside a
    /// landmass as well as between landmasses; routes around a barrier are kept. This sets which
    /// land is reached, not how far the route is: path lengths are measured between coarse cell
    /// centres, as without a crossing mode, so a route that winds inside one coarse cell, such
    /// as along a meandering corridor, counts only the distance between centres.
    Strict,
}

impl CrossingMode {
    /// Parse `"landmass"` or `"strict"`, ignoring case and surrounding whitespace. The error is
    /// the message the Python binding raises as `ValueError`.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "landmass" => Ok(Self::Landmass),
            "strict" => Ok(Self::Strict),
            other => Err(format!("Invalid nodata_crossing '{other}'. Expected 'landmass' or 'strict'.")),
        }
    }
}

/// NoData crossing settings for [`Config::crossing`].
///
/// NoData cells (NaN condition) in gaps at most `width` kilometres wide are crossable: they are
/// traversable at traversal value `traversal`, carry no habitat area, are never destinations
/// and have no protected fraction or climate similarity. A gap's width is tested once at base
/// resolution as the closing of the land cells by a disc of diameter `width`, with distances in
/// kilometres (on geographic grids each row's cells are `cos(latitude)` narrower; on rotated
/// projected grids cells keep their size, and sheared grids return an error). It is exact
/// for straight channels along rows or columns, and also fills concave corners of a coastline
/// within about a fifth of `width` of the corner; those cells too are crossable, so a coast's
/// coarse cells cost slightly more to cross when `traversal` is low. Other NoData stays out of
/// the graph, whatever its value in [`Inputs::traversal`].
///
/// BERI: a coarse cell's climate is the mean of its land (finite condition) cells' bands, as for
/// each part of a cell holding several, so land that cannot be reached never changes a score.
/// Without a crossing mode it is the mean over every cell with climate values.
///
/// Outside the tile counts as NoData, as it is outside the graph, so the tile edge never closes a
/// gap: water between land and the edge is judged as if open water lay beyond it, and a channel
/// running to the edge opens there like a river mouth. This holds for any width; a width so
/// large for the tile's cells that the gap test's grid would be unreasonably large returns an
/// error (see [`connectivity`]).
///
/// Tiles: gaps are classified and landmasses labelled from the tile's own cells, so a tile
/// matches a whole-raster run at focal cells whose window lies at least `width` inside the tile
/// edge, and in [`CrossingMode::Landmass`] only where land joined outside the tile does not
/// matter (it counts as separate).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crossing {
    pub mode: CrossingMode,
    /// Widest gap crossed, in kilometres, finite and `>= 0`; 0 permits no crossing.
    pub width: f32,
    /// Traversal value of crossable NoData in [0, 1]: 0 costs as much as fully degraded land
    /// (path weight `max_cost`), 1 as much as intact land.
    pub traversal: f32,
}

/// Settings of one [`connectivity`] call. `Config::default()` gives the Python and R defaults,
/// with level 1 included as the core requires, except `window_size`: 3 here, 5 in both APIs.
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
    /// Positive odd window size of every level but the coarsest. The level-`L` annulus spans
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
    /// NoData crossing mode; `None` keeps the behaviour of earlier versions (see the crate
    /// docs, NoData, and [`Crossing`]).
    pub crossing: Option<Crossing>,
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
            crossing: None,
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
/// Invalid inputs are checked before any work, whatever the mask, and return an error with
/// one of these messages. The Python binding raises errors as
/// `RuntimeError("Connectivity failed: <message>")`.
/// - A mask, traversal or transgrid whose rows and columns differ from the condition array's,
///   or a transgrid whose band count differs from the first one's, e.g.
///   `"Dimension mismatch for mask: (12, 14) vs (13, 14)"`,
///   `"Dimension mismatch for key 1: (12, 14) vs (13, 14)"` for traversal and
///   `"Dimension mismatch for transgrid 1: (13, 14, 3) vs (13, 14, 2)"`; transgrids without
///   bands (`"Transgrids must have at least one band"`).
/// - A level that is not a power of two (`"Level 3 is not a power of 2"`), no level 1
///   (`"Base condition level not found!"`), or levels that skip a power of two
///   (`"levels must form a continuous power-of-two sequence"`).
/// - No level-1 transform (`"Missing base-level transform (key=1)."`) or no transform for
///   another level (`"Missing level in Affine set."`).
/// - A `window_size` below 1 (`"Window size must be positive"`) or even
///   (`"Window size must be odd-numbered"`), or an even `outer_window >= window_size`
///   (`"Outer window size must be odd-numbered"`).
/// - A crossing width that is negative or not finite
///   (`"Crossing width must be a finite number of kilometres >= 0, got -1"`) or a crossing
///   traversal outside [0, 1] (`"Crossing traversal must be in [0, 1], got 2"`).
/// - Under a crossing mode, a base transform whose gap widths cannot be measured along rows and
///   columns: zero or non-finite cell sizes
///   (`"NoData crossing needs finite, non-zero cell sizes in the base transform"`), a sheared
///   grid (`"NoData crossing needs perpendicular rows and columns, but the base transform is
///   sheared: (a, b, d, e) = (1000, 300, 0, -1000)"`) or a geographic grid that is not north-up
///   (`"NoData crossing needs a north-up geographic grid (transform terms b and d of 0), got
///   b = 300, d = 0"`). Rotated projected grids are supported.
///
/// In [`CrossingMode::Strict`], an overview cell holding more than 65,534 separate pieces of
/// land and crossable NoData, which only cells of level 512 and above can, returns an error
/// while the call is prepared, before any focal cell: `"Strict NoData crossing allows at most
/// 65534 separate pieces of land and crossable NoData in one overview cell, but a level-512 cell
/// holds more; use levels up to 256"`.
///
/// Under a crossing mode, the gap test sets the tile in a NoData border more than half the width
/// deep. When that grid would exceed both 4,194,304 cells and 16 times the tile, as for a width of
/// kilometres on a small tile of sub-metre cells, the call returns an error while it is
/// prepared: `"NoData crossing width 5 km needs a NoData border of 8335 rows and 8335 columns
/// around the 20 x 20 tile for the gap test, too large a grid; use a smaller width or a larger
/// tile"`.
///
/// An integer path cost beyond `u32::MAX`, e.g. from a very large `max_cost` or unscaled
/// traversal values, stops the work on every thread and returns
/// `"Path cost overflow; check max_cost and resistance scaling!"`.
pub fn connectivity(inputs: &Inputs, config: &Config) -> anyhow::Result<Array2<f32>> {
    core::conn(inputs, config)
}
