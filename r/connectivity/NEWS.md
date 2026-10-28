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
