# connectivity-core

[![Rust core](https://github.com/csiro/connectivity/actions/workflows/core.yml/badge.svg?branch=main)](https://github.com/csiro/connectivity/actions/workflows/core.yml)
[![crates.io version](https://img.shields.io/crates/v/connectivity-core)](https://crates.io/crates/connectivity-core)
[![API documentation](https://img.shields.io/docsrs/connectivity-core)](https://docs.rs/connectivity-core)

This Rust library implements an efficient multi-resolution least-cost algorithm underlying
connectivity-based biodiversity indicators, including **Connected Habitat**
(connectivity-adjusted condition), the **Protected Area Connectedness Index
(PARC-connectedness)**, and the **Bioclimatic Ecosystem Resilience Index (BERI)**. It
combines habitat condition, least-cost paths and dispersal kernels to estimate how much
surrounding habitat is accessible across dispersal scales. The algorithm retains fine
spatial resolution near each focal cell and progressively coarsens with distance, enabling
analysis of large landscapes. PARC-connectedness incorporates protected-area coverage, while
BERI uses current and future climate-related compositional similarity to assess ecosystem
resilience under climate change. See [Valavi et al. (2026)](https://doi.org/10.32942/X2S68V)
for the full method.

The library has no Python, R, GDAL or file I/O dependencies and operates directly on
in-memory `ndarray` arrays. It also computes the fraction of each pixel covered by polygons,
for example to build protected-area proportion rasters. It provides the shared numerical
engine for the R package
[`connectivity`](https://github.com/csiro/connectivity/tree/main/r/connectivity) and the
Python package [`eco-connectivity`](https://pypi.org/project/eco-connectivity/), which
handle spatial data preparation, tiling and post-processing, including the derivation of
Connected Habitat.

```toml
[dependencies]
connectivity-core = "0.1"
```

It requires Rust 1.86 or later.

```rust
use connectivity_core::ndarray::Array2;
use connectivity_core::{connectivity, level_transforms, Config, Inputs};

let condition = Array2::from_shape_fn((24, 24), |(i, j)| ((i * 7 + j * 3) % 10) as f32 / 10.0);
let mask = condition.mapv(f32::is_nan);
let levels = [1, 2, 4];
// rasterio/affine (a, b, c, d, e, f) order; projected grids in metres
let transforms = level_transforms((100.0, 0.0, 500_000.0, 0.0, -100.0, 6_000_000.0), &levels, (0, 0));
let inputs = Inputs { condition: &condition, mask: &mask, traversal: None, transgrids: None, transforms: &transforms };
let config = Config { levels: &levels, outer_window: 5, ..Config::default() };
let connectedness = connectivity(&inputs, &config)?;
```

## Public API

- `connectivity(&Inputs, &Config)`: connectedness, or BERI when `Inputs::transgrids` holds the
  current climate followed by future scenarios (each `(rows, cols, bands)`). Returns an array
  with the condition array's shape, NaN where the mask is `true`.
- `Inputs`: borrowed condition, mask, optional traversal and transgrids, and level transforms.
- `Config`: levels, lambdas (km), `is_geo`, `max_cost`, window sizes, global tile offsets,
  window mode and PARC `pa_to_pa` gating. `Config::default()` matches the Python defaults.
- `pixel_coverage(&[Polygon], &Affine, nrows, ncols)`: fraction of each pixel covered by
  polygons with holes.
- `Affine` and `level_transforms`: (row, col) to world transforms and the per-level transforms
  of a tile, anchored to the full raster's grid.
- `WindowMode`: circular or square annuli.

The crate documentation covers array shapes, mask semantics, BERI scenario and band order,
coordinate units, tile offsets, the preparation and post-processing left to callers, and the
error and panic conditions. Work runs on the current Rayon thread pool; results are
bit-identical for any number of threads on the same build and platform.

`cargo run --release -p connectivity-core --example synthetic` runs every indicator on
synthetic arrays.

## Versioning and releases

The crate is versioned independently of the Python package (`eco-connectivity` 2.4.0 uses
`connectivity-core` 0.1.0). Core releases are tagged `connectivity-core-vX.Y.Z`; Python
releases keep `vX.Y.Z` tags, and only those publish to PyPI. Changes to the crate are listed
in [CHANGELOG.md](https://github.com/csiro/connectivity/blob/main/crates/connectivity-core/CHANGELOG.md).

## License

MIT
