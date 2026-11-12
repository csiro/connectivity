# connectivity (development version)

* `connectedness()` and `beri()` gain a `band` argument that selects one layer
  of a multi-layer `condition_file`, such as one year of a condition time
  series, for filenames and `SpatRaster` objects alike. A multi-layer
  `pa_file` or `resistance_file` uses the same layer, and a single-layer one is
  used as it is. The `beri()` climate rasters keep all their layers. Without
  `band`, the error for a multi-layer input now reads `<name> must have
  exactly one layer; choose one with band`, and a `band` outside a raster's
  layers raises an error. Only the selected layer is read, and results without
  `band` are unchanged.
* `connectedness()` and `beri()` write a run log beside `filename`: a small
  TOML file of the run's indicator, settings, inputs and output, `result.tif`
  getting `result.toml`, in the format the Python package and Eco-Connectivity
  Studio write. Dropping a log on Studio's window restores its settings. Logs
  are on by default: `save_log = FALSE` turns them off for one call,
  `options(connectivity.save_log = FALSE)` for a session, and the environment
  variable `CONNECTIVITY_SAVE_LOG` set to `0`, `false`, `no` or `off` for a
  job, in the Python package as well. A `SpatRaster` input is logged by its
  source file, and a layer of a multi-layer file also by its band; one held in
  memory is logged as `<in memory>`. An existing run log is replaced, but
  another file of that name is kept, with a warning. Not writing a log is a
  warning, never an error, and in-memory results (`filename = ""`) never
  write one.
* Reworded the package description, the README, the vignette and the help for
  `connectedness()`; results are unchanged.
* Builds against `connectivity-core` 0.2.0, which checks its inputs before any
  work. Results and error messages are unchanged.

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
