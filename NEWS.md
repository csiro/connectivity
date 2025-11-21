# Version 0.2.2
* Removed the dependency on GDAL, replacing it with `rsterio`.
* Updated logic to ignore overview level 1 (both when reading and generating) since it corresponds to the original resolution and requires no additional computation.
* Fixed the `read_raster()` and `create_overviews()` functions to ensure that the correct overview layers are created and read properly.

# Version 0.2.1
* Added Euclidean and Haversine distances for projected and unprojected grids, respectively, instead of using cell units.
* Some changes in the arguments names including `max_cost`, `window_size` and `outer_window`.
* Some small fixes for graph calculation.
