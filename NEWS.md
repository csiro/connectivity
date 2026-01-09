# Version 0.4.1
* Fixed a bug involving a mismatch between conditions and transgrids for BERI by handling NANs.
* Replaced `Arc<T>` with `Rc<T>` reference counting, since atomic operations and thread sharing are not needed.
* Improved code readability by adding more structs.

# Version 0.4.0
* Added PARC-connectedness calculation to the `connectedness()` function.
* Fixed a bug in reading transgrids with a polygon mask.

# Version 0.3.0
* Fixed the code to accept raster integer data type.
* Added a scaling factor (`scale` arg) to convert condition data to the 0–1 range.
* The output array/map is now cropped to the supplied polygon mask.
* Added type hints to the main functions.
* Made several improvements on the Rust side, such as using `Arc<T>` for cheap cloning and introducing the `Graph` struct for more organised code.

# Version 0.2.3
* Fixed isolated pixel calculation using their own values for a synthetic neighbour.
* Increased the capacity of the graph to take more neighbours for high-resolution rasters.
* Changed the default value of the `sigma` parameter to 1.0.

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
