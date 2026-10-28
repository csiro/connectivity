#' Calculate the Bioclimatic Ecosystem Resilience Index (BERI)
#'
#' @description
#' The Bioclimatic Ecosystem Resilience Index (BERI) estimates the capacity of
#' each cell to retain its biological diversity under climate change. It
#' measures how much habitat the cell can reach through the landscape that is
#' expected to support ecological communities similar to its present-day
#' communities, under the current climate and under future climate scenarios.
#' As in [connectedness()], reachable habitat is weighted by condition and
#' discounted by least-cost distance, here with a kernel representing many
#' generations of dispersal over a 50-year horizon.
#'
#' Compositional similarity comes from generalised dissimilarity modelling
#' (GDM). Each climate raster holds environmental layers transformed by a fitted
#' GDM (transgrids), with the current climate first. Scenario results are
#' combined so that a poor outcome under any one scenario lowers the index.
#' Habitat condition is used directly, without the Connected Habitat
#' transformation.
#'
#' @details
#' For a focal cell \eqn{i}, scenario \eqn{k} (\eqn{k = 0} for the current
#' climate) and dispersal bandwidth \eqn{\lambda}{lambda} (km), the accessible,
#' compositionally similar habitat is
#' \deqn{B_{ik} = \sum_j K(D_{ij}^{adj})\, h_j \tilde{A}_j s_{ijk}, \qquad K(D) = \exp\left(-\frac{(D / \lambda)^2}{283.465}\right),}{B_ik = sum_j K(D_ij(adj)) h_j A_j s_ijk,  K(D) = exp(-(D / lambda)^2 / 283.465),}
#' with distances, condition and effective areas as in [connectedness()]. The
#' Gaussian kernel approximates 50 generations of exponential dispersal
#' (\eqn{283.465 = 5.785 \times 49}{283.465 = 5.785 x 49}). The similarity of
#' destination \eqn{j} under scenario \eqn{k} to the focal cell under the
#' current climate is
#' \eqn{s_{ijk} = \exp(-\sum_b |x_{ib}^{(0)} - x_{jb}^{(k)}|)}{s_ijk = exp(-sum_b |x_ib(0) - x_jb(k)|)}
#' over the transformed layers \eqn{b}. With \eqn{m} future scenarios,
#' \deqn{\mathrm{BERI}_i = \frac{\frac{1}{2}\left(\frac{1}{m + 1}\sum_{k=0}^{m} B_{ik} + \min_k B_{ik}\right)}{\sum_j K(D_{ij}^{int})\, \tilde{A}_j s_{ij0}},}{BERI_i = [(mean_k B_ik + min_k B_ik) / 2] / [sum_j K(D_ij(int)) A_j s_ij0],}
#' the average of the mean and the worst scenario, relative to an intact
#' landscape under the current climate. Scores are averaged over `lambdas`.
#'
#' @param current_file Current-climate `terra::SpatRaster` or raster filename
#'   of GDM-transformed environmental layers (transgrids). One or more layers
#'   are supported.
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
#'   and raster alignment requirements; [make_tiles()] for large rasters.
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
#' condition_file <- system.file("extdata", "site_condition.tif", package = "connectivity")
#' climate_dir <- system.file("extdata", "transgrids", package = "connectivity")
#' current_file <- file.path(climate_dir, "1990.tif")
#' future_files <- file.path(climate_dir, c("IPS50_45.tif", "GFD50_85.tif"))
#'
#' \donttest{
#' # BERI with the current climate and two future scenarios
#' beri_map <- beri(
#'     condition_file = condition_file,
#'     current_file = current_file,
#'     future_files = future_files,
#'     lambdas = c(2, 20, 200),
#'     max_cost = 2,
#'     window_size = 5,
#'     outer_window = 9,
#'     levels = c(2, 4, 8, 16, 32),
#'     window_mode = "circular",
#'     n_threads = 2
#' )
#'
#' terra::plot(beri_map, main = "BERI")
#' }
beri <- function(condition_file,
                 current_file,
                 future_files = NULL,
                 polygon_mask = NULL,
                 closed_border = FALSE,
                 lambdas = c(2, 20, 200),
                 max_cost = 2,
                 window_size = 5L,
                 outer_window = 9L,
                 window_mode = "circular",
                 levels = c(2, 4, 8, 16, 32),
                 scale = NULL,
                 resistance_file = NULL,
                 resistance_scale = NULL,
                 n_threads = NULL,
                 filename = "",
                 overwrite = FALSE,
                 wopt = list())
{
    settings <- .normalise_settings(levels, lambdas, max_cost, window_size,
                                    outer_window, window_mode, n_threads)
    .check_scale(scale)
    prepared <- .prepare_landscape(condition_file, polygon_mask, closed_border, settings, scale)
    
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
    
    traversal <- .make_traversal(prepared$condition, resistance_file, resistance_scale, prepared)
    score <- .run_connectivity(prepared, settings, traversal, transgrids = transgrids)
    
    return(
        .raster_result(
            score, prepared$template, "beri", filename, overwrite, wopt,
            prepared$polygons, prepared$output_extent
        )
    )
}
