# Python workflow reference values

`python-workflows.csv` contains frozen float32 cell values from the existing
Python `connectivity` workflows (2.4.0, Rust core 0.1.0), using one worker,
`levels = [2, 4]`, `outer_window = 5`, `lambdas = [2, 20]` and `max_cost = 20`.
The `row` and `col` columns use R's one-based indexing. Missing and
polygon-masked cells are `NA`.

Input rasters and polygons are reconstructed by `helper-raster-fixture.R`:
13 x 14 cells for whole-raster cases and 93 x 101 for polygon cases, with
1000 m cells, EPSG:3857 and affine `(1000, 0, 500000, 0, -1000, 6000000)`.
The polygon cases exercise non-zero global offsets, an irregular boundary and
a hole. Coverage cases include overlapping features and a polygon with a hole.
Geographic cases use EPSG:4326, affine `(1/120, 0, 144, 0, -1/120, -40)` and
the same pixel-relative polygon. They check latitude-dependent distances/areas
and the default planar pixel margin. BERI is also checked with resistance.

These are end-to-end workflow references, covering raster preparation and
habitat post-processing as well as the shared Rust engine. Tests compare all
cells and missing-value locations with absolute tolerance 1e-6. Python is not
needed to run the R tests. The existing Python package and Rust core were not
modified to produce these references.
