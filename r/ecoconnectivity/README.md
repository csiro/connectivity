# ecoconnectivity

Development R interface to the existing `connectivity-core` Rust engine in
`csiro/connectivity`. The R package has an independent version and dependency
lock. Python and the Rust core retain their current layout and release process.

## Initial interface

`connectivity_matrix()` calculates raw connectedness or BERI on in-memory
matrices and arrays, with native Rust bindings and no Python runtime dependency.
It supports circular/square windows, masks, optional traversal, multi-band BERI
scenarios, global tile offsets and a per-call worker pool.

```r
library(ecoconnectivity)

condition <- matrix(seq(0.1, 1, length.out = 72), nrow = 8, ncol = 9)
score <- connectivity_matrix(
  condition,
  transform = c(1000, 0, 500000, 0, -1000, 6000000),
  levels = c(1, 2), outer_window = 5, n_threads = 1
)
```

Rows run top to bottom; columns run left to right. Transforms use
rasterio/affine order `(a, b, c, d, e, f)`, not GDAL order. Projected coordinates
are in metres, geographic coordinates in degrees, and `lambdas` in kilometres.
For a tile, use the full raster transform and zero-based `(row, column)` offsets.
NA/NaN condition cells are excluded; missing climate bands have zero similarity.

For BERI, pass `transgrids = list(current, future1, future2)`, where each array
has dimensions `(rows, columns, bands)` and identical band order. For PARC, the
caller must prepare `condition = pmax(condition, pa)` and a mask excluding
non-protected or missing condition cells. Set `pa_to_pa = TRUE` to gate
destinations to protected cells. The interface expects already scaled inputs.

Raw connectedness corresponds to the Python connectedness calculation with
`option = 1` when all preparation and settings match. Connected Habitat can be
formed as `score * condition` or `sqrt(score * condition)` for habitat results.
Do not apply this post-processing to PARC or BERI results.

Raster file/`terra` integration, automatic scaling, protected-area preparation,
polygon handling, tile planning and the higher-level Python-style API remain
future work. This package is not yet a CRAN release.

## Build and check

Source installation needs R >= 4.2, Rust >= 1.86, Cargo, GNU make, and xz/tar.
Windows additionally needs a matching Rtools installation and Rust GNU target.
Development preparation uses `rextendr`; tests use `testthat`, and documentation
uses `roxygen2`.

From the repository root:

```r
install.packages(c("rextendr", "testthat", "roxygen2"))
```

```sh
Rscript r/ecoconnectivity/tools/build-source.R /tmp/ecoconnectivity-build
R CMD INSTALL /tmp/ecoconnectivity-build/ecoconnectivity_0.0.0.9000.tar.gz
R CMD check --no-manual /tmp/ecoconnectivity-build/ecoconnectivity_0.0.0.9000.tar.gz
```

The build tool copies the existing core into an ignored, generated directory
inside the R package and resolves its workspace dependency declarations. It
vendors the R binding's locked registry dependencies into `vendor.tar.xz`.
The resulting R source archive includes everything needed for offline Cargo
compilation outside this repository. No engine source is maintained twice;
rerun preparation after changing the original core. Only development preparation
requires access to the Rust dependency registry.

For development in the checkout, run preparation before loading the package:

```r
source("r/ecoconnectivity/tools/prepare-source.R")
prepare_source("r/ecoconnectivity")
devtools::load_all("r/ecoconnectivity")
```

R regression tests reproduce the asymmetric multi-level fixture and frozen
samples in the existing Rust public API and Python binding tests. They cover
connectedness, prepared PARC, BERI, masks, matrix/band order and thread counts.
These establish engine/binding parity, not complete raster-workflow parity.

## Development conventions

- Keep all numerical algorithms in `crates/connectivity-core`.
- Put R input preparation and user-facing validation in `R/`.
- Keep native conversion code under `src/rust/src/` and avoid the R API on workers.
- Regenerate `R/extendr-wrappers.R` through the native build, then run
  `roxygen2::roxygenise("r/ecoconnectivity")` for `NAMESPACE` and `man/`.
- Keep versions independent; this development package uses core `0.1.0`.
- R checks build source archives and publish nothing.

Build scaffolding follows [rextendr](https://extendr.rs/rextendr/articles/package.html).
