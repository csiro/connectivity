# Version 0.3.0
## Added
- NoData crossing modes, set with the new `Config::crossing` field and the `Crossing` and `CrossingMode` types. NoData cells in gaps at most `Crossing::width` kilometres wide are crossed at traversal value `Crossing::traversal` and add no habitat area. `CrossingMode::Landmass` links separate landmasses only across such gaps and keeps the graph as before within a landmass; `CrossingMode::Strict` crosses no wider gap anywhere, including rivers and inlets inside one landmass. Both work on tiles with offsets, geographic grids, with PARC gating and for BERI. Under a crossing mode, coarse cells' climate is averaged over their land cells only, as each part's is, so land that cannot be reached never changes a BERI score; traversal values on NoData that may not be crossed are ignored. Outside the tile counts as NoData in the gap test, so the tile edge never closes a gap. `CrossingMode::parse()` reads `"landmass"` and `"strict"`.
- Invalid crossing settings return errors before any work: `"Crossing width must be a finite number of kilometres >= 0, got <width>"` and `"Crossing traversal must be in [0, 1], got <value>"`. Gap widths are measured along the grid's rows and columns, so rotated projected grids are supported, while a sheared grid, a geographic grid that is not north-up, or zero or non-finite cell sizes return errors under a crossing mode (`"NoData crossing needs perpendicular rows and columns, ..."`, `"NoData crossing needs a north-up geographic grid ..."`, `"NoData crossing needs finite, non-zero cell sizes in the base transform"`). In strict mode, an overview cell holding more than 65,534 separate pieces, possible only at levels of 512 and above, returns an error (`"Strict NoData crossing allows at most 65534 separate pieces ..."`) before any focal cell is computed.

## Changed
- Breaking: `Config` has a new field, `crossing`; code that builds `Config` without `..Config::default()` must set it. `Config::default()` sets it to `None`, which keeps the results of 0.2.1 bit for bit.
- The documentation of `Inputs::condition` no longer says that NaN cells are left out of the graph. Without a crossing mode they are left out at level 1, but coarser levels link land across narrow NoData gaps; the crate docs describe this under NoData.

# Version 0.2.1
## Added
- Added `VERSION`, the crate's version as a string, so that callers can record which engine produced a result, as the run logs of the Python and R packages do. Results are unchanged.

# Version 0.2.0
## Changed
- `connectivity()` now checks shapes, band counts, levels, transforms and window sizes before any work, whatever the mask, and returns an error instead of panicking. The messages are unchanged, except that a level that is not a power of two no longer starts with `Failed average resampling.:`. Levels that skip a power of two, a missing level 1 or level transform and even window sizes used to pass when every cell was masked, and skipped levels also passed when the levels past the gap had no window cells; they now fail in every case.
- A path cost beyond `u32::MAX` now stops the work on every thread and returns an error instead of panicking on a worker thread. The message is unchanged.
- Valid inputs give bit-identical results.

## Fixed
- A mask or transgrid whose rows and columns differ from the condition array's, and transgrids with different band counts, now return errors (`Dimension mismatch for mask: ...`, `Dimension mismatch for transgrid <k>: ...`). They used to panic on indexing or, for band counts, run without an error and give wrong results. A non-positive `window_size` and transgrids without bands also return errors.

# Version 0.1.0
## Added
- First release of the numerical engine of the `eco-connectivity` Python package as a standalone crate: habitat and PARC connectedness, the Bioclimatic Ecosystem Resilience Index (BERI) and fractional pixel coverage of polygons on in-memory `ndarray` arrays of one raster tile, with no Python, GDAL or file I/O dependencies. `eco-connectivity` 2.4.0 calls it through a thin PyO3 wrapper; earlier changes to the engine are listed under the Python package's versions in the repository's [NEWS.md](https://github.com/csiro/connectivity/blob/main/python/NEWS.md).
- Added `connectivity()`, which takes borrowed `Inputs` (condition, mask, optional traversal and BERI transgrids, and level transforms) and `Config` settings, with `Config::default()` matching the Python defaults. Also added `pixel_coverage()` for polygons with holes, `Affine` and `level_transforms()` for tile grids anchored to the full raster, and `WindowMode` with `WindowMode::parse()`.
- Work runs on the current Rayon thread pool, and outputs are bit-identical for any thread count on the same build and platform. The crate re-exports `ndarray` (0.17) so callers can build matching arrays, and requires Rust 1.86 or later.
- Documented the method, array shapes, mask semantics, units, tile offsets, the preparation left to callers, and the error and panic conditions in the crate docs, with a tested example and `examples/synthetic.rs`, which runs every indicator on synthetic arrays.
