# connectivity: multi-resolution landscape connectivity in Python

[![Build](https://github.com/csiro/connectivity/actions/workflows/publish.yml/badge.svg?branch=main)](https://github.com/csiro/connectivity/actions/workflows/publish.yml)
[![CodeQL](https://github.com/csiro/connectivity/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/csiro/connectivity/actions/workflows/github-code-scanning/codeql)
[![Python versions](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://pypi.org/project/eco-connectivity/)
[![PyPI version](https://img.shields.io/pypi/v/eco-connectivity)](https://pypi.org/project/eco-connectivity/)
[![Downloads](https://api.pepy.tech/badge/eco-connectivity)](https://pepy.tech/projects/eco-connectivity)
[![DOI](https://img.shields.io/badge/DOI-10.32942%2FX2S68V-blue)](https://doi.org/10.32942/X2S68V)

- [connectivity: multi-resolution landscape connectivity in Python](#connectivity-multi-resolution-landscape-connectivity-in-python)
  - [Installation](#installation)
    - [Install from PyPI](#install-from-pypi)
    - [Install from Source](#install-from-source)
    - [Rust Core Library](#rust-core-library)
  - [Connectivity Analysis](#connectivity-analysis)
    - [Connected Habitat (Connectedness)](#connected-habitat-connectedness)
      - [Optional resistance surface](#optional-resistance-surface)
    - [PARC-Connectedness](#parc-connectedness)
    - [Bioclimatic Ecosystem Resilience Index (BERI)](#bioclimatic-ecosystem-resilience-index-beri)
  - [Running Analysis with Tiles](#running-analysis-with-tiles)
  - [Citation and scientific references](#citation-and-scientific-references)
  - [Acknowledgements and licence](#acknowledgements-and-licence)
      
A Python package for calculating **Connected Habitat** (a.k.a. connectivity-adjusted condition), the **Protected Area Connectedness Index (PARC-connectedness)** and the **Bioclimatic Ecosystem Resilience Index (BERI)**. It combines habitat condition, least-cost paths and dispersal kernels to estimate how much surrounding habitat is accessible across dispersal scales. Fine spatial detail is retained near each focal cell, while progressively coarser grids represent more distant areas, making large-landscape analyses efficient. PARC-connectedness incorporates protected-area coverage; BERI incorporates current and future climate-related compositional similarity to assess ecosystem resilience under climate change. The package uses `rasterio` and `geopandas` for spatial input and output and the [`connectivity-core`](https://docs.rs/connectivity-core) Rust engine, which it shares with the R package [`connectivity`](https://github.com/csiro/connectivity/tree/main/r/connectivity). See [Valavi et al. (2026)](https://doi.org/10.32942/X2S68V) for the full method.

## Installation
### Install from PyPI
The package is published on PyPI as `eco-connectivity` (NOTE: `pip install connectivity` installs an unrelated package):

```bash
pip install eco-connectivity
```

The import name is still `connectivity`:

```python
from connectivity import connectedness, beri
```

### Install from Source

To build the latest development version from the [GitHub repository](https://github.com/csiro/connectivity), install [Rust](https://www.rust-lang.org/tools/install) first, then run:

```bash
git clone https://github.com/csiro/connectivity.git
cd connectivity
pip install ./python
```

### Rust Core Library

The numerical engine is a separate Rust crate, [`connectivity-core`](https://github.com/csiro/connectivity/tree/main/crates/connectivity-core), which works on in-memory arrays without Python, GDAL or file I/O. Rust programs can use it directly; see its [README](https://github.com/csiro/connectivity/blob/main/crates/connectivity-core/README.md) and crate documentation. `cargo test --workspace` runs every Rust test, and `cargo test -p connectivity-core` needs no Python.

## Connectivity Analysis
Resolve the bundled example paths once so they can be reused throughout the
analysis:

```python
from connectivity import beri, connectedness, example_data_path, resolution_info

condition_file = example_data_path("site_condition")
pa_file = example_data_path("pa_proportion")
current_file = example_data_path("transgrids/1990")
future_files = [
    example_data_path("transgrids/IPS50_45"),
    example_data_path("transgrids/GFD50_85"),
]
```

To bound the spatial extent of the connectivity calculation, the algorithm limits how far it searches across neighboring cells in the condition raster. The approximate one-sided search reach is computed as:
`max_reach = outer_window * max(levels) * resolution`. For example, with a 1 km resolution raster, a max-level of 32, and an `outer_window` of 11, the resulting search reach is:
      
```text
distance = outer_window × max_level × resolution
         = 11 × 32 × 1 km
         = 352 km  
```    

Use `resolution_info()` to preview how candidate `levels` will change raster
dimensions and approximate search reach before running the analysis:

```python
resolution_info(condition_file, outer_window=9)
```

```text
File: .../connectivity/data/site_condition.tif
Base raster: 588 x 516 cells (303.41K total)
Base resolution: 0.008333 degrees
CRS: EPSG:4326
Bands: 1
Data type(s): float32
Nodata: -9999.0
Outer window: 9
Small-dimension warning threshold: < 16 cells
Distance note: approximate km values use centre latitude -41.550

Candidate internally generated levels:
  Level    Width   Height      Cells   % base           Resolution            Reach     Reach km  Note
      1      588      516    303.41K 100.000%     0.008333 degrees    0.075 degrees     8.349 km
      2      294      258     75.85K  25.000%      0.01667 degrees     0.15 degrees      16.7 km
      4      147      129     18.96K   6.250%      0.03333 degrees      0.3 degrees      33.4 km
      8       74       65      4.81K   1.585%      0.06667 degrees      0.6 degrees     66.79 km
     16       37       33      1.22K   0.402%       0.1333 degrees      1.2 degrees     133.6 km
     32       19       17        323   0.106%       0.2667 degrees      2.4 degrees     267.2 km
     64       10        9         90   0.030%       0.5333 degrees      4.8 degrees     534.3 km  small grid; review
    128        5        5         25   0.008%        1.067 degrees      9.6 degrees     1.07K km  small grid; review
    256        3        3          9   0.003%        2.133 degrees     19.2 degrees     2.14K km  small grid; review
    512        2        2          4   0.001%        4.267 degrees     38.4 degrees     4.27K km  small grid; review
```

The default `window_mode="circular"` uses source-centered circular annuli with
fractional area/count support at annulus boundaries. The
`window_mode="square"` option uses the same source-centered fractional
construction with square annuli. These modes change indicator values, so
compare outputs only between runs that use the same window mode.

<img src="https://raw.githubusercontent.com/csiro/connectivity/main/figs/circular_multires_windows.png" alt="Circular multi-resolution windows" width="900">

Each coloured neighbourhood represents a different raster aggregation level.
Fine levels capture nearby cells at higher resolution, while coarser levels
extend the search over larger distances. Their contributions are combined into
a single graph, in which the least-cost path from the focal cell is calculated
for each aggregated cell.
See [Valavi et al. (2026)](https://doi.org/10.32942/X2S68V) for the full method.

### Connected Habitat (Connectedness)

To compute connected-habitat (or plain connectedness), you only need a habitat condition raster. Use `option` argument to generate the connected-habitat from connectedness and input condition with:    
1. `connectedness`
2. `connectedness * condition`
3. `sqrt(connectedness * condition)` — geometric mean (default)

```python
connd = connectedness(
    condition_file = condition_file,
    lambdas = [2, 20, 200],
    max_cost = 2.0, 
    window_size = 5, 
    outer_window = 11,
    window_mode = "circular",
    levels = [2, 4, 8, 16, 32], 
    option = 3,
    filename = "./results/connected_habitat.tif"
)
```

![](https://raw.githubusercontent.com/csiro/connectivity/main/figs/condition.png)

#### Optional resistance surface

By default the habitat condition raster does double duty: it weights how costly each cell is to
move through **and** supplies the habitat value used in the indicator. Pass an optional
`resistance_file` to **decouple** these two roles. The resistance raster (values in `[0, 1]`,
higher = harder to cross, scaled with `resistance_scale`) then drives **only** the least-cost path
traversal, while condition still supplies the habitat value:

```python
connd = connectedness(
    condition_file = condition_file,
    resistance_file = resistance_file,   # optional; decoupled movement-cost surface
    resistance_scale = None,             # divide resistance into [0, 1] if needed
    max_cost = 2.0,
    window_mode = "circular",
    levels = [2, 4, 8, 16, 32],
)
```

The edge weight becomes `w = (1.0 - max_cost) * (1 - resistance) + max_cost`, so `resistance = 0`
is free (`w = 1`) and `resistance = 1` is the most costly (`w = max_cost`). When `resistance_file`
is omitted, condition is used for traversal as before, and the output is unchanged. Cells valid in
condition but missing a resistance value fall back to condition for traversal and raise a warning,
so the analysis domain is never changed silently. The same `resistance_file` / `resistance_scale`
arguments are available on `beri()`.

### PARC-Connectedness
To compute PARC-connectedness, provide both:
* a habitat condition raster, and
* a protected-areas proportion raster.    
     
When `pa_file` (proportion of protected-areas in each cell) is supplied, the function automatically returns PARC-connectedness instead of standard connectedness.

```python
parcc = connectedness(
    condition_file = condition_file,
    pa_file = pa_file,
    lambdas = [2, 20, 200],
    max_cost = 2.0, 
    window_size = 5, 
    outer_window = 11,
    window_mode = "circular",
    levels = [2, 4, 8, 16, 32], 
    filename = "./results/parc_connectedness.tif"
)
```

<img src="https://raw.githubusercontent.com/csiro/connectivity/main/figs/parc.png" width="385" height="300">

Use `pixel_coverage()` when you need the proportion of each raster pixel covered
by polygon geometry. The calculation is backed by a performant Rust
implementation:

```python
from connectivity import pixel_coverage

coverage = pixel_coverage(
    "./data/polygons.gpkg",
    condition_file,
)
```

### Bioclimatic Ecosystem Resilience Index (BERI)
To compute BERI, you must provide:
* a condition raster
* current GDM transgrids
* one or more future transgrids scenarios (as a list)

```python
beris = beri(
    condition_file = condition_file,
    current_file = current_file,
    future_files = future_files,
    lambdas = [2, 20, 200], 
    max_cost = 2.0, 
    window_size = 5, 
    outer_window = 11,
    window_mode = "circular",
    levels = [2, 4, 8, 16, 32],
    filename = "./results/berri.tif"
)
```
<img src="https://raw.githubusercontent.com/csiro/connectivity/main/figs/beri.png" width="385" height="300">


## Running Analysis with Tiles

To run the model using tiles, create a rectangular tile polygon as a GeoDataFrame
and pass it to the `polygon_mask` argument. This limits data loading to only the
portion required for the tile, i.e. the output core plus the internally buffered
neighborhood.

Use `make_tile()` to generate non-overlapping output-core tiles whose internal
boundaries align with the coarsest aggregation level. This is preferred over
creating overlap-expanded tile polygons externally. Be sure to set
`closed_border = False` (the default) so that neighborhood information is
included around each tile core.

```python
from connectivity import connectedness, make_tile

levels = [2, 4, 8, 16, 32]

tile_id = 0
tile_poly = make_tile(
    raster_file = condition_file,
    nrows = 4,
    ncols = 4,
    tile_id = tile_id,
    align_to = max(levels),
)

connd = connectedness(
    condition_file = condition_file,
    polygon_mask = tile_poly,
    closed_border = False,
    margin_px = 32,
    lambdas = [2, 20, 200],
    max_cost = 2.0, 
    window_size = 5, 
    outer_window = 11,
    window_mode = "circular",
    levels = levels, 
    option = 1,
    filename = f"./results/connected_habitat_tile_{tile_id}.tif"
)
```

For large, uneven workloads, use balanced tiles:

```python
tile_poly = make_tile(
    raster_file = condition_file,
    nrows = 4,
    ncols = 4,
    tile_id = tile_id,
    align_to = max(levels),
    balanced = True,
    io_weight = 0.1,
)
```

<img src="https://raw.githubusercontent.com/csiro/connectivity/main/figs/balanced_tiles.png" width="500" height="440">

Balanced tiling uses a deterministic raster-mask cost estimate while preserving
the same `align_to` boundary alignment. It is useful for global rasters where
some tiles contain mostly ocean or nodata and others contain many valid land
pixels. Use `balanced = False` (the default) when equal aligned tiles are enough.
When `balanced = False`, `io_weight` is ignored.

## Citation and scientific references

To cite the `connectivity` library in publications and reports, please use:

Valavi, R., Mokany, K., Ware, C., Vickers, M., Giljohann, K. M., & Ferrier,
S. (2026). **A scalable multi-resolution framework for connectivity-based
biodiversity indicators**. *EcoEvoRxiv*, preprint.
[doi:10.32942/X2S68V](https://doi.org/10.32942/X2S68V).

Underlying methods:

- Drielsma, M., Ferrier, S., & Manion, G. (2007). A raster-based technique
  for analysing habitat configuration: The cost–benefit approach.
  *Ecological Modelling*, 202(3–4), 324–332.
  [doi:10.1016/j.ecolmodel.2006.10.016](https://doi.org/10.1016/j.ecolmodel.2006.10.016).
- Drielsma, M. J., Love, J., Taylor, S., Thapa, R., & Williams, K. J. (2022).
  General Landscape Connectivity Model (GLCM): a new way to map whole of
  landscape biodiversity functional connectivity for operational planning
  and reporting. *Ecological Modelling*, 465, 109858.
  [doi:10.1016/j.ecolmodel.2021.109858](https://doi.org/10.1016/j.ecolmodel.2021.109858).
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
[MIT licence](https://github.com/csiro/connectivity/blob/main/python/LICENSE).
The Python binding uses `PyO3`/`maturin` for its interface and `rasterio`,
`geopandas` and `shapely` for its spatial workflows.

The bundled Tasmania data are licensed separately under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Please cite:

Ware, C., Valavi, R., Vickers, M., Giljohann, K. M., Mokany, K., Purvis, A.,
Walkden, P., De Palma, A., Duffin, C., Contu, S., Harwood, T., Hoskins, A., &
Ferrier, S. (2026). *Global biodiversity indicator data for BHI, BERI,
PARC-representativeness, PARC-connectedness and ecosystem condition (2000–2024)*.
Version 4. CSIRO Data Collection.
[doi:10.25919/3aka-y730](https://doi.org/10.25919/3aka-y730).

See the [example-data README](https://github.com/csiro/connectivity/blob/main/python/connectivity/data/README.md)
for layer metadata and attribution, also installed with the package as
`connectivity/data/README.md`.

[Back to top!](#connectivity-multi-resolution-landscape-connectivity-in-python)
