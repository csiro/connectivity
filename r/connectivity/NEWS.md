# connectivity 0.0.0.9000

* Bundled the Python package's Tasmania example rasters under `inst/extdata`,
  with a vignette calculating and plotting all four indicators in memory.
* Depends on the published `connectivity-core` crate from crates.io, with locked
  dependencies vendored for offline source installation.
* Added terra-based `connectedness()`, `beri()` and `pixel_coverage()` raster
  interfaces, including scaling, resistance, polygon boundaries and file output.
* Added frozen Python workflow comparisons for complete raster results.
* Included the shared Python/Rust repository MIT licence verbatim in LICENSE.md.
* Internal matrix interface to `connectivity-core` 0.1.0, including connectedness,
  caller-prepared PARC inputs, BERI arrays, traversal, masks and tile offsets.
* Uses native Rust bindings with no Python runtime dependency.
