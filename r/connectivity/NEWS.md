# connectivity (development version)

## Breaking changes

* `connectedness()` and `beri()` now default to `nodata_crossing = "strict"`,
  `crossing_width = 0.5` km and `crossing_cost = 4`, as the Python package
  3.0. This changes the calculation near NoData: version 0.1.0 could link land
  across wider gaps at no cost at coarser levels, creating steps on islands,
  coastlines and irregular boundaries.
  * Pass `nodata_crossing = NULL` to reproduce results of 0.1.0. To forbid all
    gap crossings while keeping strict routing, set `crossing_width = 0`.
  * The default width and cost are provisional assumptions, not calibrated
    values. Set a wider gap explicitly when testing sensitivity, including in
    BERI applications.
  * On the bundled Tasmania raster (cells about 0.69 x 0.93 km), the new
    default permits no gap crossings. Plain connectedness changes in almost
    every valid cell, by 0.003 on average and up to 0.21; 4% of cells change
    by more than 0.01.
* Every argument of `connectedness()` and `beri()` after `condition_file` must
  be named, in full, as Python's keyword-only arguments: they follow `...`,
  so a positional, misspelt or abbreviated argument raises `Arguments after
  condition_file must be named; unknown: ...` instead of being matched or
  ignored. Pass the required `current_file` to `beri()` by name; leaving it out
  raises `current_file is required`.
* The arguments of `connectedness()` and `beri()` are reordered as in Python
  3.0: inputs (`band`, `scale`, `pa_file`, `pa_to_pa`, `option` or
  `current_file`, `future_files`, then `resistance_file`, `resistance_scale`),
  method (`lambdas`, `max_cost`, `levels`, `window_size`, `outer_window`,
  `window_mode`), area of interest (`polygon_mask`, `closed_border`), NoData
  crossing, then execution and output (`n_threads`, `filename`, `overwrite`,
  `wopt`, `save_log`).

## New features

* NoData crossing controls in `connectedness()` and `beri()`:
  * `nodata_crossing = "strict"` limits crossings to gaps within
    `crossing_width` at every level. `"landmass"` applies that limit between
    separate landmasses and keeps coarse-level links across wider gaps within a
    landmass. `NULL` is the legacy handling of 0.1.0.
  * `crossing_width` is a finite distance of at least 0 km. Crossed NoData
    contributes movement cost and carries no habitat area, protected area or
    climate. Cells beyond the raster edge count as NoData, so the edge cannot
    close a gap.
  * `crossing_cost` is a finite cost per unit distance relative to intact land,
    at least 1: 4 costs four times as much as intact land. It is independent of
    condition, resistance and `max_cost`, and can exceed `max_cost`. Coarse
    cells with land and crossable NoData use their mean cost.
  * With a crossing mode, `beri()` averages climate over land cells only:
    climate values on NoData and on unreachable land are excluded from scores.
  * Open-border `polygon_mask` runs, such as tiles, read further by
    `crossing_width`, so adjacent tiles still join without seams.
  * Invalid settings raise errors with the Python package's messages. Under
    `nodata_crossing = NULL`, a changed width or cost is ignored with a message.
  * Grids may be north-up geographic or projected; rotated grids are still
    rejected.
* `connectedness()` and `beri()` gain a `band` argument that selects one layer
  of a multi-layer `condition_file`, such as one year of a condition time
  series, for filenames and `SpatRaster` objects alike. `band` selects a layer
  of `condition_file` only: `pa_file` and `resistance_file` must have one
  layer, which is used with every layer of the condition raster, and a
  multi-layer one raises `<name> must have exactly one layer, but it has <n>;
  band selects a layer of condition_file only, so write the layer to use to a
  single-band raster first`. The `beri()` climate rasters keep all their
  layers. Without `band`, a multi-layer `condition_file` raises
  `condition_file must have exactly one layer; choose one with band`, and a
  `band` outside its layers raises an error. Only the selected layer is read.
* `connectedness()` and `beri()` write a run log beside `filename`: a small
  TOML file of the run's indicator, settings, inputs and output, `result.tif`
  getting `result.toml`, in format 2, which the Python package 3.0 also
  writes. It always records the NoData crossing: `nodata_crossing` is
  `"strict"`, `"landmass"` or `"none"` (`NULL`), with `crossing_width_km` and
  `crossing_cost` under a mode. Dropping a log on Eco-Connectivity Studio's
  window restores its settings; Studio 0.2 reads format 1 only and refuses
  these logs until it supports format 2. Logs
  are on by default: `save_log = FALSE` turns them off for one call,
  `options(connectivity.save_log = FALSE)` for a session, and the environment
  variable `CONNECTIVITY_SAVE_LOG` set to `0`, `false`, `no` or `off` for a
  job, in the Python package as well. A `SpatRaster` input is logged by its
  source file, and a layer of a multi-layer file also by its band; one held in
  memory is logged as `<in memory>`. An existing run log is replaced, but
  another file of that name is kept, with a warning. Not writing a log is a
  warning, never an error, and in-memory results (`filename = ""`) never
  write one.

## Minor improvements

* Reworded the package description, the README, the vignette and the help for
  `connectedness()`; results are unchanged.
* Builds against `connectivity-core` 0.3.0, which checks its inputs before any
  work and adds the NoData crossing.

# connectivity 0.1.0

* First release.
* `connectedness()` calculates Connected Habitat, raw connectedness and
  PARC-connectedness, with optional scaling, resistance surfaces, polygon
  boundaries and file output.
* `beri()` calculates the Bioclimatic Ecosystem Resilience Index from current
  climate and optional future scenarios.
* `pixel_coverage()` calculates exact fractional polygon coverage of raster
  cells.
* `resolution_info()` previews aggregation dimensions, resolutions, cell counts,
  approximate search reach and small-grid flags.
* `connectedness()` and `beri()` use `window_size = 5` and `outer_window = 9`
  by default.
* `make_tiles()` and `make_tile()` split large rasters into aligned, optionally
  cost-balanced tiles. Merged open-border tiles equal a whole-raster result.
* Example rasters for Tasmania and a vignette calculating all four indicators.
* Uses the `connectivity-core` Rust engine, with its dependencies vendored for
  offline source installation.
