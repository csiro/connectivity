# Version 1.1.1
## Fixed
- Filtered bounded outputs from `connectedness()` and `beri()` are now clipped back to the theoretical `[0, 1]` range after FFT notch filtering. This applies to connectedness (`option = 1`), connected habitat (`option = 2`), geometric-mean connected habitat (`option = 3`), PARC-connectedness, and BERI whenever filtering is enabled.

## Changed
- `remove_grid_effect()` now runs FFT on the finite-data bounding box and uses reflect-padding (`fft_pad_px`) to reduce extent-dependent periodic artifacts.

# Version 1.1.0
## Added
- Added `remove_grid_effect()` as a standalone FFT notch-filter post-processing function (also exported from the top-level package import).
- Added a Rust `inpaint_nans_diffusion` implementation (Rayon-parallel), with optional thread control via `n_threads`.

## Fixed
- Fixed seam artifacts caused by swapped row/column arguments in graph distance coordinate conversion (`Affine::xy` now consistently receives `(row, col)`).
- Fixed overview alignment across independently processed tiles by anchoring aggregation windows to global dataset pixel coordinates via tile offsets.
- Fixed cross-level edge-neighbour mapping to be offset-aware (global-anchor consistent) when linking a level to its higher level.
- Fixed occasional missing edge rows/cols in buffered tile reads by adding a half-pixel epsilon to the buffer extent.

## Changed
- Graph construction now passes tile offsets through the full path (`core -> builder -> graph::fringe`) to keep multi-resolution linking globally aligned.
- Node IDs are now tile-invariant (`NodeId = u64` from level/global-row/global-col), removing tile-local ID drift between adjacent runs.
- Core traversal is now deterministic where relevant (sorted overview levels and sorted Dijkstra targets before accumulation).
- Non-closed tile reads are snapped to the coarsest overview grid, improving consistency of aggregated counts/weights at tile boundaries.
- Added `Area` resampling for 2D overviews (latitude-weighted by `cos(phi)`), alongside `Average`, `Sum`, and `Count`.
- 3D overview generation now explicitly supports only `Average` and `Sum` resampling.
- `overview_info()` now reports estimated overview dimensions and does not require pre-existing file overviews.
- `connectedness()` and `beri()` now apply FFT notch filtering (`remove_grid_effect`) instead of the previous Gaussian smoother.
- Filtering in `connectedness()` and `beri()` is now controlled via `filter_kwargs`:
  pass `None` to disable filtering, or `{}` to apply default filtering (`notch_width=3`).
- Inpainting used by filtering now relies on the Rust implementation and uses `n_threads` when provided.

## Removed
- Removed legacy `smoothing_filter()` backward-compatibility wrapper.
- Removed SciPy-based inpainting fallback and dropped the `scipy` Python dependency.

# Version 1.0.0
## Added
- Incorporated habitat area into connectivity calculations using per-node cell counts (aggregate-based weighting).
- Integrated overview generation directly in Rust to eliminate preprocessing steps and reduce raster I/O bottlenecks.
- Added the `anyhow` crate for improved, centralized error handling.
- Refactored core functionality into a dedicated module for improved structure and maintainability.
## Removed
- Removed `create_overviews()` and `has_overview()`; overview handling is now fully internal.

# Version 0.6.1
* Fixed a bug for reading files with different overview levels.
* Added additional early checks for grid shapes and values.

# Version 0.6.0
* Added `closed_border` argument to control whether cells outside the mask are excluded from the analysis.
* Fixed the polygon-masked output to not return the buffered array.
* Added compilation CPU optimisation flags (slower complie but faster run).
* Changed the `overview_levels` argument of the `create_overviews()` function to `levels` for consistency. Also, `file_path` in `read_raster()` function.

# Version 0.5.0
* Added `has_overview()` function for checking overviews.

# Version 0.4.2
* A few internal structural changes; e.g., changes to the NumPy array memory layout and added structure in the Rust code for performance improvement.

# Version 0.4.1
* Fixed a bug involving a mismatch between conditions and transgrids for BERI by handling NANs.
* Replaced `Arc<T>` with `Rc<T>` reference counting, since atomic operations and thread sharing are not needed.
* Improved code readability by adding more structs.

# Version 0.4.0
* Added PARC-connectedness calculation to the `connectedness()` function.
* Fixed a bug in reading transgrids with a polygon mask.

# Version 0.3.0
* Fixed the code to accept raster integer data type.
* Added a scaling factor (`scale` arg) to convert condition data to the 0–1 range.
* The output array/map is now cropped to the supplied polygon mask.
* Added type hints to the main functions.
* Made several improvements on the Rust side, such as using `Arc<T>` for cheap cloning and introducing the `Graph` struct for more organised code.

# Version 0.2.3
* Fixed isolated pixel calculation using their own values for a synthetic neighbour.
* Increased the capacity of the graph to take more neighbours for high-resolution rasters.
* Changed the default value of the `sigma` parameter to 1.0.

# Version 0.2.2
* Removed the dependency on GDAL, replacing it with `rasterio`.
* Updated logic to ignore overview level 1 (both when reading and generating) since it corresponds to the original resolution and requires no additional computation.
* Fixed the `read_raster()` and `create_overviews()` functions to ensure that the correct overview layers are created and read properly.

# Version 0.2.1
* Added Euclidean and Haversine distances for projected and unprojected grids, respectively, instead of using cell units.
* Some changes in the arguments names including `max_cost`, `window_size` and `outer_window`.
* Some small fixes for graph calculation.

# Version 0.2.0
Added geographic distance (in km) instead of cell units
Replaced the scale argument with max_cost
Various minor improvements

# Version 0.1.0
* Initial version.
