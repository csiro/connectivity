# Version 0.2.3
* Fixed the isolated pixel calculations by using its values for a synthetic neighbour.
* Increased the capacity of graph to take more neighbours for high resoltion rasters.
* Changed the default value of `sigma` paramter to 1.0.

# Version 0.2.2
* Removed the dependency on GDAL, replacing it with `rasterio`.
* Updated logic to ignore overview level 1 (both when reading and generating) since it corresponds to the original resolution and requires no additional computation.
* Fixed the `read_raster()` and `create_overviews()` functions to ensure that the correct overview layers are created and read properly.

# Version 0.2.1
* Added Euclidean and Haversine distances for projected and unprojected grids, respectively, instead of using cell units.
* Some changes in the arguments names including `max_cost`, `window_size` and `outer_window`.
* Some small fixes for graph calculation.

# Version 0.2.0
Added geographic distance (in km) instead of cell units
Replaced the scale argument with max_cost
Various minor improvements

# Version 0.1.0
* Initial version.
