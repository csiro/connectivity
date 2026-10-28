#' Calculate Connected Habitat or PARC connectedness
#'
#' Reads rasters with [terra::rast()], prepares condition and optional protected
#' area and resistance inputs, and returns a georeferenced raster.
#' The defaults match the Python package's scientific
#' settings, including the Connected Habitat square-root transformation.
#'
#' @param condition_file Single-layer `terra::SpatRaster` or raster filename.
#'   Habitat condition must be in `[0, 1]` after scaling. Missing values are
#'   excluded. All raster inputs must share a CRS, resolution and pixel grid;
#'   auxiliary rasters must cover the analysis window. They are not resampled.
#' @param pa_file Optional single-layer protected-area proportion raster or
#'   filename. Values must be in `[0, 1]` after scaling. The analysis uses the
#'   maximum of condition and protected fraction and evaluates protected cells
#'   only. PARC output is raw connectedness, independent of `option`.
#' @param pa_to_pa Restrict PARC destinations to protected cells, weighted by
#'   protected fraction at coarser levels. Ignored if `pa_file` is `NULL`.
#' @param polygon_mask Optional polygon `terra::SpatVector`, vector filename or
#'   object accepted by [terra::vect()]. Polygons are reprojected to the raster
#'   CRS. Both polygon and raster must have a CRS, or both omit one. The result
#'   is cropped to the intersecting polygon bounds and cells outside polygons
#'   are `NA`. Cells touched by a polygon are included.
#' @param closed_border If `FALSE`, retain surrounding landscape context for
#'   the analysis before cropping the output. If `TRUE`, exclude the landscape
#'   outside `polygon_mask` from both paths and habitat contributions.
#' @param margin_px Extra valid focal-cell margin around polygons in open-border
#'   analyses, in pixels. Must not exceed `(outer_window + 3) * max(levels)`.
#'   Ignored without a polygon or with closed borders.
#' @param lambdas Positive dispersal bandwidths in kilometres. Scores are
#'   averaged across bandwidths.
#' @param max_cost Finite relative traversal cost, at least 1.
#' @param window_size Positive odd window size for non-coarsest levels.
#' @param outer_window Positive odd coarsest window size, at least `window_size`.
#' @param window_mode Either `"circular"` (default) or `"square"`.
#' @param levels Aggregation levels. Positive integers are rounded to the
#'   nearest power of two, with ties rounded down; level 1 is included. The
#'   result must be a continuous power-of-two sequence. Overviews are computed
#'   by Rust from base-resolution values; external raster overviews are unused.
#' @param scale Optional divisor for condition values (`NULL`, 0 or 1 leaves
#'   values unchanged). A numeric pair or list supplies separate condition and
#'   protected-area divisors, e.g. `c(100, 100)` or `list(NULL, 100)`.
#' @param resistance_file Optional single-layer resistance raster or filename.
#'   After scaling, `1 - resistance` is clipped to `[0, 1]` for path traversal.
#'   Missing resistance at valid condition cells falls back to condition and
#'   raises a warning. Resistance does not alter habitat values or the domain.
#' @param resistance_scale Optional divisor applied only to resistance values.
#' @param option Habitat output: 1 returns connectedness, 2 multiplies by
#'   condition, and 3 (default) takes the square root of that product.
#' @param n_threads Optional positive integer number of worker threads.
#'   `NULL` uses Rayon's default thread count.
#' @param filename Optional output filename. An empty string returns an
#'   in-memory raster. Files default to single-precision floating-point values.
#' @param overwrite Whether an existing output file may be overwritten.
#' @param wopt Named list of [terra::writeRaster()] options.
#' @return A single-layer `terra::SpatRaster`, optionally written to `filename`.
#'   Missing and masked cells are `NA`. Computation and habitat post-processing
#'   use single precision, matching the Python interface.
#' @details Projected CRS units are converted to metres for the Rust engine.
#'   Geographic coordinates use degrees. Dispersal bandwidths are kilometres.
#'   An `outer_window` smaller than `window_size` is increased to match it.
#'   A missing raster CRS triggers a warning and the Python reader's coordinate
#'   heuristic (degrees for sub-degree cells, otherwise metres where clear).
#'
#'   Each call reads the complete analysis window into memory. For large areas,
#'   call the function with polygon tiles and open borders: the read window is
#'   padded and aligned to the full raster's coarsest aggregation grid so that
#'   adjacent results use the same overview cells.
#' @export
#' @examples
#' habitat <- terra::rast(nrows = 8, ncols = 9, xmin = 0, xmax = 9000,
#'                        ymin = 0, ymax = 8000, crs = "EPSG:3857")
#' terra::values(habitat) <- seq(0.1, 1, length.out = 72)
#' score <- connectedness(habitat, levels = c(2, 4), outer_window = 5,
#'                        n_threads = 1)
connectedness <- function(
    condition_file, pa_file = NULL, pa_to_pa = FALSE, polygon_mask = NULL,
    closed_border = FALSE, margin_px = 32L, lambdas = c(2, 20, 200),
    max_cost = 2, window_size = 3L, outer_window = 9L,
    window_mode = "circular", levels = c(2, 4, 8, 16, 32), scale = NULL,
    resistance_file = NULL, resistance_scale = NULL, option = 3L,
    n_threads = NULL, filename = "", overwrite = FALSE, wopt = list()) {
  settings <- .normalise_settings(levels, lambdas, max_cost, window_size,
                                  outer_window, window_mode, n_threads)
  .check_integers(option, "option", length = 1L, minimum = 1L)
  if (!option %in% 1:3) stop("option must be one of 1, 2 or 3", call. = FALSE)
  .check_flag(pa_to_pa, "pa_to_pa")
  if (pa_to_pa && is.null(pa_file)) {
    message("pa_to_pa is ignored because pa_file was not supplied")
    pa_to_pa <- FALSE
  }
  scales <- .condition_scales(scale)
  prepared <- .prepare_landscape(condition_file, polygon_mask, closed_border,
                                 margin_px, settings, scales[[1L]])
  name <- if (option == 1L) "connectedness" else "connected_habitat"
  if (!is.null(pa_file)) {
    pa <- .read_window(pa_file, prepared, "pa_file", scale = scales[[2L]])
    .check_matrix(pa, "pa_file", unit_interval = TRUE)
    prepared$condition <- pmax(prepared$condition, pa)
    prepared$mask <- !(pa > 0 & !is.na(prepared$condition))
    prepared$mask[is.na(prepared$mask)] <- TRUE
    name <- "parc_connectedness"
  }
  traversal <- .make_traversal(prepared$condition, resistance_file,
                               resistance_scale, prepared)
  score <- .run_connectivity(prepared, settings, traversal, pa_to_pa = pa_to_pa)
  if (is.null(pa_file) && option != 1L) {
    score <- .float32(score * prepared$condition)
    if (option == 3L) score <- .float32(sqrt(score))
  }
  .raster_result(score, prepared$template, name, filename, overwrite, wopt,
                  prepared$polygons, prepared$output_extent)
}
