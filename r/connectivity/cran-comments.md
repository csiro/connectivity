## Test environments

* Local: macOS 27.0 (aarch64), R 4.6.1, rustc 1.99.1.

## R CMD check results

0 errors | 1 warning | 1 note

* The warning only reports that the `checkbashisms` script is not installed
  locally. `configure`, `configure.win`, `cleanup` and `cleanup.win` are
  single-command POSIX `sh` scripts.
* This is a new submission.
* The tarball is about 35 MB. Of this, 32 MB are the example rasters in
  `inst/extdata`, used by the vignette to reproduce the published Tasmania
  example of the companion Python package with all four indicators. The
  GeoTIFFs are already DEFLATE-compressed; lossless recompression saves less
  than 2%. We would welcome advice if a separate data package is preferred.

## Example data licence

The package code is MIT-licensed. The bundled example rasters are published
by CSIRO under CC BY-NC-SA 4.0 (Ware et al. 2026,
<https://doi.org/10.25919/3aka-y730>). This is declared in the `Description`
field and documented in `inst/extdata/README.md`. The code does not depend on
the data.

## Rust

* All crates are vendored in `src/rust/vendor.tar.xz` (xz) and built offline
  with `cargo build -j 2 --offline`. The `rustc` version is reported in the
  installation log, and `SystemRequirements` declares Cargo and rustc >= 1.86.
* Authors and licences of the vendored crates are listed in `LICENSE.note`,
  referenced from the `Copyright` field.
