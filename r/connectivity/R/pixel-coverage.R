#' Calculate fractional polygon coverage of raster cells
#'
#' Unions polygon geometries using `terra`, then calls the existing Rust core's
#' exact per-cell polygon clipping algorithm. Holes and multipart polygons are
#' preserved. This does not use sub-cell sampling or rasterize's approximate
#' coverage option.
#'
#' @param polygon Polygon `terra::SpatVector`, vector filename, or object
#'   accepted by [terra::vect()]. Geometries are reprojected to the raster CRS
#'   where necessary and unioned so overlapping polygons are not counted twice.
#' @param raster_file Reference `terra::SpatRaster` or raster filename. Only its
#'   grid and CRS are used; values and nodata do not mask the coverage result.
#' @inheritParams connectedness
#' @return A single-layer `terra::SpatRaster` of fractions in `[0, 1]` matching
#'   the full reference grid. Empty polygon input produces zero coverage.
#'   Fractions refer to area in the raster's coordinate plane.
#' @export
#' @examples
#' grid <- terra::rast(nrows = 2, ncols = 3, xmin = 0, xmax = 30,
#'                     ymin = 0, ymax = 20, crs = "EPSG:3857")
#' polygon <- terra::as.polygons(terra::ext(5, 20, 2, 18), crs = terra::crs(grid))
#' fraction <- pixel_coverage(polygon, grid, n_threads = 1)
pixel_coverage <- function(polygon, raster_file, n_threads = NULL,
                           filename = "", overwrite = FALSE, wopt = list()) {
    reference <- .as_raster(raster_file, "raster_file", single = FALSE, values = FALSE)
    polygons <- .as_polygons(polygon, reference, "polygon")
    if (!is.null(n_threads)) .check_integers(n_threads, "n_threads", length = 1L, minimum = 1L)
    rings <- list()
    if (nrow(polygons) && nrow(terra::geom(polygons))) {
        polygons <- terra::aggregate(polygons, dissolve = TRUE)
        geometry <- terra::geom(polygons)
        parts <- split(seq_len(nrow(geometry)), paste(geometry[, "geom"], geometry[, "part"], sep = "/"))
        rings <- lapply(parts, function(indices) {
            part <- geometry[indices, , drop = FALSE]
            exterior <- part[part[, "hole"] == 0, c("x", "y"), drop = FALSE]
            holes <- sort(unique(part[part[, "hole"] != 0, "hole"]))
            return(c(list(as.double(t(exterior))), lapply(holes, function(hole) {
                return(as.double(t(part[part[, "hole"] == hole, c("x", "y"), drop = FALSE])))
            })))
        })
    }
    result <- pixel_coverage_impl(unname(rings), .affine(reference),
                                  as.integer(c(nrow(reference), ncol(reference))),
                                  if (is.null(n_threads)) 0L else as.integer(n_threads))
    if (!is.null(result$err)) stop(as.character(result$err), call. = FALSE)
    score <- matrix(result$ok, nrow(reference), ncol(reference))
    return(.raster_result(score, reference, "pixel_coverage", filename, overwrite, wopt))
}
