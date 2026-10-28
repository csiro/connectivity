#' Calculate the Bioclimatic Ecosystem Resilience Index
#'
#' Reads condition and climate rasters with `terra` and calculates BERI using
#' current climate first and future scenarios in the supplied order. Condition
#' is used directly; no Connected Habitat multiplication or square-root
#' post-processing is applied.
#'
#' @param current_file Current-climate `terra::SpatRaster` or raster filename.
#'   Layers are climate-composition bands. One or more bands are supported.
#' @param future_files Optional character vector of raster filenames, list of
#'   raster objects/filenames, or a single `SpatRaster` for one future scenario.
#'   Every scenario must have the same number and order of bands as current
#'   climate. `NULL` or an empty list computes the current-climate index only.
#'   Missing climate bands give the cell a similarity of zero.
#' @param scale Optional condition divisor. Climate bands are not scaled.
#' @inheritParams connectedness
#' @return A single-layer `terra::SpatRaster`, optionally written to `filename`.
#'   Its values are BERI scores; missing and polygon-masked cells are `NA`.
#' @seealso [connectedness()] for input units, polygon boundaries, memory use
#'   and raster alignment requirements.
#' @references
#' Ferrier, S., Harwood, T. D., Ware, C., and Hoskins, A. J. (2020).
#' A globally applicable indicator of the capacity of terrestrial ecosystems
#' to retain biological diversity under climate change: The bioclimatic
#' ecosystem resilience index. Ecological Indicators, 117, 106554.
#' \doi{10.1016/j.ecolind.2020.106554}. Introduces BERI.
#'
#' Harwood, T., Love, J., Drielsma, M., Brandon, C., and Ferrier, S. (2022).
#' Staying connected: assessing the capacity of landscapes to retain
#' biodiversity in a changing climate. Landscape Ecology, 37, 3123--3139.
#' \doi{10.1007/s10980-022-01534-5}. Develops the spatial resilience and
#' connectedness approach.
#'
#' Valavi, R., Mokany, K., Ware, C., Vickers, M., Giljohann, K. M., and
#' Ferrier, S. (2026). A scalable multi-resolution framework for
#' connectivity-based biodiversity indicators. EcoEvoRxiv, preprint.
#' \doi{10.32942/X2S68V}. Describes the multi-resolution implementation.
#' @export
#' @examples
#' habitat <- terra::rast(nrows = 8, ncols = 9, xmin = 0, xmax = 9000,
#'                        ymin = 0, ymax = 8000, crs = "EPSG:3857")
#' terra::values(habitat) <- seq(0.1, 1, length.out = 72)
#' climate <- c(habitat, 1 - habitat)
#' score <- beri(habitat, climate, future_files = list(climate * 0.8),
#'               levels = c(2, 4), outer_window = 5, n_threads = 1)
beri <- function(
    condition_file, current_file, future_files = NULL, polygon_mask = NULL,
    closed_border = FALSE, margin_px = 32L, lambdas = c(2, 20, 200),
    max_cost = 2, window_size = 3L, outer_window = 9L,
    window_mode = "circular", levels = c(2, 4, 8, 16, 32), scale = NULL,
    resistance_file = NULL, resistance_scale = NULL, n_threads = NULL,
    filename = "", overwrite = FALSE, wopt = list()) {
    settings <- .normalise_settings(levels, lambdas, max_cost, window_size,
                                    outer_window, window_mode, n_threads)
    .check_scale(scale)
    prepared <- .prepare_landscape(condition_file, polygon_mask, closed_border,
                                   margin_px, settings, scale)
    if (is.null(future_files)) future_files <- list()
    if (inherits(future_files, "SpatRaster")) future_files <- list(future_files)
    if (is.character(future_files)) future_files <- as.list(future_files)
    if (!is.list(future_files)) {
        stop("future_files must be a raster, filenames, a list of rasters or NULL", call. = FALSE)
    }
    scenarios <- c(list(current_file), unname(future_files))
    transgrids <- lapply(seq_along(scenarios), function(i) {
        return(.read_window(
            scenarios[[i]], prepared,
            if (i == 1L) "current_file" else paste0("future_files[[", i - 1L, "]]"),
            single = FALSE
        ))
    })
    bands <- vapply(transgrids, function(x) return(dim(x)[3L]), integer(1))
    if (any(bands != bands[1L])) {
        stop("Every climate scenario must have the same band count as current_file", call. = FALSE)
    }
    traversal <- .make_traversal(prepared$condition, resistance_file,
                                 resistance_scale, prepared)
    score <- .run_connectivity(prepared, settings, traversal, transgrids = transgrids)
    return(.raster_result(
        score, prepared$template, "beri", filename, overwrite, wopt,
        prepared$polygons, prepared$output_extent
    ))
}
