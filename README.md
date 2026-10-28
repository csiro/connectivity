# connectivity: a multi-resolution landscape connectivity algorithm

[![Python](https://github.com/csiro/connectivity/actions/workflows/publish.yml/badge.svg?branch=main)](https://github.com/csiro/connectivity/actions/workflows/publish.yml)
[![R](https://github.com/csiro/connectivity/actions/workflows/r-cmd-check.yml/badge.svg?branch=main)](https://github.com/csiro/connectivity/actions/workflows/r-cmd-check.yml)
[![Rust core](https://github.com/csiro/connectivity/actions/workflows/core.yml/badge.svg?branch=main)](https://github.com/csiro/connectivity/actions/workflows/core.yml)
[![PyPI version](https://img.shields.io/pypi/v/eco-connectivity)](https://pypi.org/project/eco-connectivity/)
[![crates.io](https://img.shields.io/crates/v/connectivity-core)](https://crates.io/crates/connectivity-core)
[![DOI](https://img.shields.io/badge/DOI-10.32942%2FX2S68V-blue)](https://doi.org/10.32942/X2S68V)

A multi-resolution landscape connectivity algorithm for calculating ***Habitat Connectivity (Connected-Habitat)***, ***PARC Connectedness*** and the ***Bioclimatic Ecosystem Resilience Index (BERI)***. It combines habitat condition, least-cost paths and dispersal kernels to estimate how much surrounding habitat is accessible across different dispersal scales. The algorithm retains fine spatial detail near each focal cell and represents more distant areas at progressively coarser resolutions, using raster overview layers generated on the fly to make large-landscape analyses efficient. See [Valavi et al. (2026)](https://doi.org/10.32942/X2S68V) for the full method.

## Packages

The numerical engine is written in Rust and shared by the Python and R packages. Each package has its own installation instructions, documentation and changelog:

| Package | Directory | Install |
|---|---|---|
| Python, [`eco-connectivity`](https://pypi.org/project/eco-connectivity/) (imported as `connectivity`) | [`python/`](python/) | `pip install eco-connectivity` |
| R, `connectivity` (not yet on CRAN) | [`r/connectivity/`](r/connectivity/) | Build from source; see its [README](r/connectivity/README.md#installation) |
| Rust, [`connectivity-core`](https://crates.io/crates/connectivity-core) | [`crates/connectivity-core/`](crates/connectivity-core/) | `connectivity-core = "0.1"` in `Cargo.toml` |

## Development

The repository root is a Cargo workspace containing the core crate and the PyO3 extension in `python/`. The R binding has its own Cargo workspace and lockfile, and builds against the published `connectivity-core` release.

```bash
cargo test --workspace          # every Rust test (the extension links libpython)
cargo test -p connectivity-core # the core alone, without Python
pip install ./python            # build and install the Python package
```

## Citation
To cite `connectivity` in publications and reports, please use:

Valavi, R., Mokany, K., Ware, C., Vickers, M., Giljohann, K. M., & Ferrier, S. (2026). **A scalable multi-resolution framework for connectivity-based biodiversity indicators**. *EcoEvoRxiv*. [https://doi.org/10.32942/X2S68V](https://doi.org/10.32942/X2S68V)
