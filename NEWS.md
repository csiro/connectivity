# Version 0.2.1
* Removed the dependency on GDAL (replaced with `rsterio`).
* Ignore the overview level 1, as it's the original resolution and there calculation required.

# Version 0.2.1
* Added Euclidean and Haversine distances for projected and unprojected grids, respectively, instead of using cell units.
* Some changes in the arguments names including `max_cost`, `window_size` and `outer_window`.
* Some small fixes for graph calculation.
