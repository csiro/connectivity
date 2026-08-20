# Bundled example data

Example rasters for the notebooks in `examples/`. Load them with
`connectivity.example_data_path()` rather than by relative path:

```python
from connectivity import example_data_path

condition_file = example_data_path("site_condition")
pa_file = example_data_path("pa_proportion")
baseline = example_data_path("transgrids/1990")
```

All layers share the same grid:

| Property | Value |
| --- | --- |
| Extent | Tasmania, Australia (143.7–148.6 E, 43.7–39.4 S) |
| Size | 588 x 516 cells |
| CRS | EPSG:4326 |
| Resolution | 0.008333 degrees (30 arc-second, ~1 km) |
| Type | float32, DEFLATE-compressed GeoTIFF |

## Layers

### `site_condition.tif`

Habitat condition, 1 band, values 0–1, nodata `-9999`.

- **Source:** Ware et al. (2026), CSIRO Data Collection
- **Version / date:** Version 4, published 20 March 2026; data coverage 2000–2024
- **Licence:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Attribution:** Cite Ware et al. (2026) using the full dataset citation in
  [References](#references).

### `pa_proportion.tif`

Proportion of each cell under protection, 1 band, values 0–1, nodata `-9999`.
Passed as `pa_file` to `connectedness()` to get PARC-connectedness.

- **Source:** Ware et al. (2026), CSIRO Data Collection
- **Version / date:** Version 4, published 20 March 2026; data coverage 2000–2024
- **Licence:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Attribution:** Cite Ware et al. (2026) using the full dataset citation in
  [References](#references).

### `transgrids/*.tif`

Compositional turnover grids, 19 bands each, nodata `NaN`. `1990.tif` is the current
baseline; the others are future climate scenarios used as `future_files` in `beri()`.

Scenario names below are inferred from the filenames — please confirm before publishing.

| File | Scenario |
| --- | --- |
| `1990.tif` | current baseline |
| `ACC50_85.tif` | ACCESS, 2050, RCP 8.5 |
| `GFD50_85.tif` | GFDL, 2050, RCP 8.5 |
| `IPS50_26.tif` | IPSL, 2050, RCP 2.6 |
| `IPS50_45.tif` | IPSL, 2050, RCP 4.5 |
| `IPS50_60.tif` | IPSL, 2050, RCP 6.0 |
| `IPS50_85.tif` | IPSL, 2050, RCP 8.5 |

- **Source:** Ware et al. (2026), CSIRO Data Collection
- **Version / date:** Version 4, published 20 March 2026; data coverage 2000–2024
- **Licence:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Attribution:** Cite Ware et al. (2026) using the full dataset citation in
  [References](#references).

## Using your own layers

Any raster passed alongside these must match the grid above exactly — 588 x 516
cells at the same extent and resolution. `connectedness()` raises on a shape
mismatch between `condition_file` and `pa_file`.

## References

- Ware, Chris; Valavi, Roozbeh; Vickers, Mat; Giljohann, Kate; Mokany, Karel;
  Purvis, Andy; Walkden, Patrick; De Palma, Adriana; Duffin, Connor; Contu,
  Sara; Harwood, Thomas; Hoskins, Andrew; & Ferrier, Simon (2026). *Global
  biodiversity indicator data for BHI, BERI, PARC-representativeness,
  PARC-connectedness and ecosystem condition (2000–2024).* Version 4. CSIRO.
  Data Collection. [https://doi.org/10.25919/3aka-y730](https://doi.org/10.25919/3aka-y730)

- Valavi, R., Mokany, K., Ware, C., Vickers, M., Giljohann, K. M., & Ferrier,
  S. (2026). *A scalable multi-resolution framework for connectivity-based
  biodiversity indicators.* EcoEvoRxiv.
  [Preprint](https://ecoevorxiv.org/repository/view/12146/)
