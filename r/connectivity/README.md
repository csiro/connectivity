# connectivity: multi-resolution landscape connectivity in R

[![R CMD check](https://github.com/csiro/connectivity/actions/workflows/r-cmd-check.yml/badge.svg?branch=main)](https://github.com/csiro/connectivity/actions/workflows/r-cmd-check.yml)
[![R versions](https://img.shields.io/badge/R-%E2%89%A54.2-blue)](https://www.r-project.org/)
[![DOI](https://img.shields.io/badge/DOI-10.32942%2FX2S68V-blue)](https://doi.org/10.32942/X2S68V)

An R package for calculating **Connected Habitat**, **PARC-connectedness** and
the **Bioclimatic Ecosystem Resilience Index (BERI)**. It combines habitat
condition, least-cost paths and dispersal kernels to estimate how much
surrounding habitat is accessible across dispersal scales. Fine spatial detail
is retained near each focal cell, while progressively coarser grids represent
more distant areas, making large-landscape analyses efficient.

PARC-connectedness incorporates protected-area coverage; BERI incorporates
current and future climate-related compositional similarity to assess ecosystem
resilience under climate change. These indicators support biodiversity monitoring
and conservation planning. See [Valavi et al. (2026)](https://doi.org/10.32942/X2S68V)
for the multi-resolution method.

The R interface uses `terra` for spatial input and output and the
[`connectivity-core`](https://docs.rs/connectivity-core) Rust numerical
engine.

## Key features

- `connectedness()`: raw connectedness, condition-adjusted Connected Habitat,
  and PARC-connectedness, including protected-to-protected destinations.
- `beri()`: resilience from current climate and optional future scenarios.
- `pixel_coverage()`: exact fractional polygon coverage, preserving holes and
  multipart geometries and unioning overlapping polygons.
- `resolution_info()`: preview aggregation levels, grid sizes and search reach.
- `make_tiles()` and `make_tile()`: split large rasters into aligned, optionally
  cost-balanced tiles whose merged results equal a whole-raster calculation.
- Raster filenames or `terra::SpatRaster` inputs, georeferenced raster outputs,
  optional resistance surfaces, polygon boundaries and multithreaded computation.

## Installation

This development package is not yet available on CRAN. Source installation needs
**R >= 4.2**, **Rust >= 1.86** with Cargo, GNU make, and xz/tar. Windows also
requires matching Rtools and the Rust GNU target. Building the bundled vignette
requires Pandoc, `knitr` and `rmarkdown`.

Clone the repository, or use an existing checkout:

```sh
git clone https://github.com/csiro/connectivity.git
cd connectivity
```

From the repository root, install the R dependencies, then build and install:

```r
install.packages(c("terra", "rextendr", "knitr", "rmarkdown"))
```

```sh
Rscript r/connectivity/tools/build-source.R /tmp/connectivity-r-build
R CMD INSTALL /tmp/connectivity-r-build/connectivity_*.tar.gz
```

Use a fresh output directory for the source archive. Preparation requires access
to the Rust registry or a populated Cargo cache. The build tool vendors locked
Rust dependencies, including the published `connectivity-core = '=0.1.0'`, so
the resulting archive supports offline Cargo compilation. The R package has its
own version and dependency lock; local edits to the repository's Rust core do
not affect R builds.

## Connectivity analysis

### Example data and input requirements

The package bundles nine Tasmania GeoTIFFs: habitat condition, protected-area
proportions, and current and six future climate-composition grids. Locate them
in the installed package with `system.file()`:

```r
library(connectivity)
library(terra)

data_dir <- system.file("extdata", package = "connectivity")
condition_file <- file.path(data_dir, "site_condition.tif")
pa_file <- file.path(data_dir, "pa_proportion.tif")
current_file <- file.path(data_dir, "transgrids", "1990.tif")
future_files <- file.path(
    data_dir, "transgrids", c("IPS50_45.tif", "GFD50_85.tif")
)
condition <- rast(condition_file)
```

Condition and protected-area proportions must be single-layer rasters with
values in `[0, 1]`; missing condition cells are excluded. Use `scale = 100`
for condition stored as percentages, or `scale = c(100, 100)` to scale both
condition and protected fractions in `connectedness()`. Auxiliary rasters must
share the condition raster's CRS, resolution and pixel grid and cover the
analysis window. They are not resampled automatically. Supply a valid CRS;
projected units are converted internally, and `lambdas` always uses kilometres.

### Resolution planning

```r
plan <- resolution_info(condition, outer_window = 9)
```

This prints raster metadata and candidate aggregation levels and invisibly
returns a data frame. Approximate search reach is
`outer_window * max(levels) * max(res(condition))` in CRS units; the kilometre
estimate for geographic rasters uses the centre latitude. This planning span
differs from the dispersal bandwidths in `lambdas`. Small-grid flags guide
review rather than selecting levels automatically.

Aggregation levels are generated from the base raster during each calculation.
Indicator calls include level 1 and require a continuous sequence of
powers of two after rounding supplied levels. Both window sizes must be positive
odd integers; `outer_window` is increased to `window_size` if smaller.

<img src="https://raw.githubusercontent.com/csiro/connectivity/main/figs/circular_multires_windows.png" alt="Circular multi-resolution neighbourhoods around a focal cell" width="900">

Each coloured neighbourhood represents a different aggregation level. Their
contributions form one graph for least-cost path calculations. The default
`window_mode = "circular"` uses circular annuli with fractional boundary support;
`"square"` uses square annuli. Window mode affects indicator values, so use the
same mode when comparing analyses.

### Connected Habitat (connectedness)

Only habitat condition is required. The following call shows the default
scientific settings, also used by the subsequent indicator examples:

```r
habitat <- connectedness(
    condition_file = condition,
    lambdas = c(2, 20, 200),
    max_cost = 2,
    window_size = 5,
    outer_window = 9,
    window_mode = "circular",
    levels = c(2, 4, 8, 16, 32),
    option = 3
)
plot(habitat, main = "Connected habitat")
```

Scores are averaged across dispersal bandwidths. Choose the output with `option`:

| Option | Output |
| --- | --- |
| `1` | Raw connectedness |
| `2` | `connectedness * condition` |
| `3` (default) | `sqrt(connectedness * condition)` |

Indicator functions and `pixel_coverage()` return a single-layer `SpatRaster`.
Pass `filename` to write an output, `overwrite = TRUE` to replace an existing
file, and `wopt` for `terra::writeRaster()` options. Files default to float32.
Set `n_threads` to control worker count; `NULL` uses the engine's default.

### Optional resistance surface

Condition normally supplies both habitat value and traversal costs. Provide
`resistance_file` to control movement costs separately, while retaining condition
as the habitat value. Higher resistance means harder passage. For example:

```r
# Illustrative surface; replace with an independently prepared resistance raster.
resistance <- 1 - condition
habitat_resistance <- connectedness(
    condition_file = condition,
    resistance_file = resistance
)
```

Use `resistance_scale` for a separate divisor if needed. After scaling,
`1 - resistance` is clipped to `[0, 1]` for traversal, so resistance of 1 or
more, including `Inf`, gives the highest path cost. Missing resistance at valid
condition cells falls back to condition with a warning. The same arguments are
available in `beri()`.

### PARC-connectedness

Supply a protected-area proportion raster to calculate PARC-connectedness:

```r
parc <- connectedness(condition_file, pa_file = pa_file)
plot(parc, main = "PARC-connectedness")
```

The calculation uses the maximum of condition and protected fraction, and only
cells with positive protected fractions receive scores. `pa_to_pa = TRUE`
restricts destinations to protected cells, weighted by protected fraction at
coarser levels. PARC returns raw connectedness regardless of `option`.

Use `pixel_coverage()` to prepare protected fractions from polygons. This
illustrative rectangle shares the example raster's CRS:

```r
protected_area <- as.polygons(
    ext(146, 147, -42.5, -41.5), crs = crs(condition)
)
coverage <- pixel_coverage(protected_area, condition)
```

Polygon inputs may also be vector filenames or objects accepted by
`terra::vect()`. Polygons are reprojected when necessary. Coverage fractions
refer to area in the raster coordinate plane; reference nodata does not mask
the coverage result.

### Bioclimatic Ecosystem Resilience Index (BERI)

Provide habitat condition and current transgrids from generalised dissimilarity
modelling (GDM), with optional future scenarios. Each scenario must have the same
number and order of bands as the current grid; the bundled climate rasters each
contain 19 bands.

```r
beri_map <- beri(
    condition_file = condition_file,
    current_file = current_file,
    future_files = future_files
)
plot(beri_map, main = "BERI")
```

`future_files` accepts a character vector of filenames, a list of rasters or
filenames, or one `SpatRaster` for a single scenario. The result is one BERI
raster combining the supplied scenarios, without Connected Habitat
post-processing. Omit `future_files` for a current-climate index. Climate bands
are not scaled by `scale`; missing climate bands give a cell similarity of zero.

## Polygon analyses and tiles

Pass `polygon_mask` to `connectedness()` or `beri()` to limit the output area.
With the default `closed_border = FALSE`, the package reads surrounding context,
aligns the read window to the full raster's coarsest aggregation grid, then crops
and masks the result to the polygon. Setting `closed_border = TRUE` excludes
outside habitat from both paths and contributions.

```r
region <- as.polygons(ext(146, 147, -42.5, -41.5), crs = crs(condition))
regional_habitat <- connectedness(
    condition,
    polygon_mask = region,
    closed_border = FALSE
)
```

Each call loads its analysis window into memory. For large rasters, split the
grid with `make_tiles()`, run each tile with open borders and write it to a
file, then combine the files. Supply the full condition raster so that each
tile can read its surrounding landscape; the combined tiles then equal a
whole-raster calculation:

```r
tiles <- make_tiles(condition_file, nrows = 2, ncols = 2, align_to = 32)
tile_files <- sprintf("connected_habitat_tile_%02d.tif", tiles$id)
for (i in seq_len(nrow(tiles))) {
    connectedness(condition_file, polygon_mask = tiles[i], filename = tile_files[i])
}
habitat_tiled <- vrt(tile_files)
```

`terra::vrt()` reads the tile files as one virtual raster; use
`merge(sprc(tile_files), filename = ...)` to write a single merged file.

Set `align_to` to the largest aggregation level. `balanced = TRUE` places tile
edges to balance the number of cells with data rather than tile sizes, and
`make_tile()` returns a single tile, e.g. for one job in a batch array. Tile
numbers and cell ranges are one-based.

For a full worked analysis, see the [Tasmania vignette](vignettes/tasmania-indicators.Rmd),
or open `vignette("tasmania-indicators", package = "connectivity")` after installation.

## Citation and scientific references

Use `citation("connectivity")` for the package citation:

Valavi, R., Mokany, K., Ware, C., Vickers, M., Giljohann, K. M., & Ferrier,
S. (2026). **A scalable multi-resolution framework for connectivity-based
biodiversity indicators**. *EcoEvoRxiv*, preprint.
[doi:10.32942/X2S68V](https://doi.org/10.32942/X2S68V).

Underlying methods:

- Harwood, T., Love, J., Drielsma, M., Brandon, C., & Ferrier, S. (2022).
  Staying connected: assessing the capacity of landscapes to retain
  biodiversity in a changing climate. *Landscape Ecology*, 37, 3123–3139.
  [doi:10.1007/s10980-022-01534-5](https://doi.org/10.1007/s10980-022-01534-5).
- Ferrier, S., Harwood, T. D., Ware, C., & Hoskins, A. J. (2020).
  A globally applicable indicator of the capacity of terrestrial ecosystems
  to retain biological diversity under climate change: The bioclimatic
  ecosystem resilience index. *Ecological Indicators*, 117, 106554.
  [doi:10.1016/j.ecolind.2020.106554](https://doi.org/10.1016/j.ecolind.2020.106554).

## Acknowledgements and licence

Developed at the **Commonwealth Scientific and Industrial Research Organisation
(CSIRO)**. Copyright © 2026 CSIRO. The code is distributed under the
[MIT licence](LICENSE.md). The R binding uses
`extendr`/`rextendr` and `terra` for its interface and spatial workflows. The
authors and licences of the vendored Rust crates are listed in
[`LICENSE.note`](LICENSE.note).

The bundled Tasmania data are licensed separately under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Please cite:

Ware, C., Valavi, R., Vickers, M., Giljohann, K. M., Mokany, K., Purvis, A.,
Walkden, P., De Palma, A., Duffin, C., Contu, S., Harwood, T., Hoskins, A., &
Ferrier, S. (2026). *Global biodiversity indicator data for BHI, BERI,
PARC-representativeness, PARC-connectedness and ecosystem condition (2000–2024)*.
Version 4. CSIRO Data Collection.
[doi:10.25919/3aka-y730](https://doi.org/10.25919/3aka-y730).

See the [example-data README](inst/extdata/README.md) for layer metadata and
attribution, also installed at `file.path(data_dir, "README.md")`.
