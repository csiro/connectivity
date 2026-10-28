#' Calculate habitat connectedness, Connected Habitat or PARC-connectedness
#'
#' @description
#' Habitat connectedness measures how much of the potentially connected habitat
#' around each cell is retained, relative to a hypothetical intact landscape.
#' Surrounding habitat counts more when it is in better condition and closer
#' along its least-cost path, which lengthens through degraded habitat. Values
#' range from 0 (isolated) to 1 (as connected as an intact landscape).
#'
#' By default, the function returns Connected Habitat (a.k.a.
#' connectivity-adjusted condition): the geometric mean of a cell's own habitat
#' condition and its connectedness, used as an input to the Biodiversity
#' Habitat Index (BHI). With `pa_file`, it returns the Protected Area
#' Connectedness Index (PARC-connectedness) instead: how well protected cells
#' are connected to other protected areas and to surrounding intact habitat.
#'
#' Each cell's neighbourhood keeps fine detail nearby and represents more
#' distant landscape with progressively coarser aggregated cells (Valavi et al.
#' 2026). Inputs are raster filenames or `terra` SpatRaster objects, and the
#' result is a georeferenced raster.
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
#' @param lambdas Positive dispersal bandwidths in kilometres. Scores are
#'   averaged across bandwidths.
#' @param max_cost Finite relative traversal cost, at least 1.
#' @param window_size Positive odd window size for non-coarsest levels.
#' @param outer_window Positive odd coarsest window size, at least `window_size`.
#' @param window_mode Either `"circular"` (default) or `"square"`.
#' @param levels Aggregation levels. Positive integers are rounded to the
#'   nearest power of two, with ties rounded down; level 1 is included. The
#'   result must be a continuous power-of-two sequence. Aggregated grids are
#'   computed from the base-resolution values.
#' @param scale Optional divisor for condition values (`NULL`, 0 or 1 leaves
#'   values unchanged). A numeric pair or list supplies separate condition and
#'   protected-area divisors, e.g. `c(100, 100)` or `list(NULL, 100)`.
#' @param resistance_file Optional single-layer resistance raster or filename.
#'   After scaling, `1 - resistance` is clipped to `[0, 1]` for path traversal,
#'   so resistance of 1 or more, including `Inf`, gives the highest path cost.
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
#'   use single precision.
#' @details
#' For a focal cell \eqn{i} and one dispersal bandwidth \eqn{\lambda}{lambda}
#' (km), connectedness is
#' \deqn{C_i = \frac{\sum_j \exp(-D_{ij}^{adj} / \lambda)\, h_j \tilde{A}_j}{\sum_j \exp(-D_{ij}^{int} / \lambda)\, \tilde{A}_j},}{C_i = [sum_j exp(-D_ij(adj) / lambda) h_j A_j] / [sum_j exp(-D_ij(int) / lambda) A_j],}
#' where \eqn{j} runs over the cells of the multi-resolution neighbourhood,
#' \eqn{h_j} is habitat condition and \eqn{\tilde{A}_j}{A_j} is effective area.
#' \eqn{D_{ij}^{adj}}{D_ij(adj)} is the least-cost distance when crossing a cell
#' of condition \eqn{h} costs its length times \eqn{(1 - \mu) h + \mu}{(1 - mu) h + mu},
#' with \eqn{\mu}{mu} = `max_cost`, and \eqn{D_{ij}^{int}}{D_ij(int)} is the
#' distance through an intact landscape. With `resistance_file`, the cost factor
#' is \eqn{1 + (\mu - 1) r}{1 + (mu - 1) r} for resistance \eqn{r}, while
#' \eqn{h_j} remains the destination weight. Scores are averaged over `lambdas`.
#' `option = 2` returns \eqn{h_i C_i} and `option = 3` returns Connected
#' Habitat, \eqn{\sqrt{h_i C_i}}{sqrt(h_i C_i)}.
#'
#' PARC-connectedness replaces \eqn{h} with
#' \eqn{\varphi = \max(h, p)}{phi = max(h, p)}, for protected fraction \eqn{p},
#' in both path costs and destination weights, and scores only cells with
#' \eqn{p > 0}. Paths still cross unprotected land. With `pa_to_pa = TRUE`, each
#' destination in the numerator is also weighted by its protected share, so only
#' protected destinations count.
#'
#' Projected CRS units are converted to metres for the Rust engine.
#'   Geographic coordinates use degrees. Dispersal bandwidths are kilometres.
#'   An `outer_window` smaller than `window_size` is increased to match it.
#'   A missing raster CRS triggers a warning. Coordinates are then assumed to be
#'   degrees for cells smaller than one unit and metres for larger cells; mixed
#'   cases are judged from whether the extent fits longitude/latitude bounds.
#'
#'   Each call reads the complete analysis window into memory. For large areas,
#'   call the function for each tile from [make_tiles()] with open borders and
#'   merge the results with [terra::merge()]. The read window is padded by
#'   `(outer_window + 3) * max(levels)` pixels and aligned to the full raster's
#'   coarsest aggregation grid, so merged tiles equal a whole-raster result.
#' @seealso [make_tiles()] to split large rasters into tiles.
#' @references
#' Valavi, R., Mokany, K., Ware, C., Vickers, M., Giljohann, K. M., and
#' Ferrier, S. (2026). A scalable multi-resolution framework for
#' connectivity-based biodiversity indicators. EcoEvoRxiv, preprint.
#' \doi{10.32942/X2S68V}. Describes the multi-resolution implementation.
#'
#' Harwood, T., Love, J., Drielsma, M., Brandon, C., and Ferrier, S. (2022).
#' Staying connected: assessing the capacity of landscapes to retain
#' biodiversity in a changing climate. Landscape Ecology, 37, 3123--3139.
#' \doi{10.1007/s10980-022-01534-5}. Describes the underlying landscape
#' connectedness approach.
#'
#' Ferrier, S., Harwood, T. D., Ware, C., and Hoskins, A. J. (2020).
#' A globally applicable indicator of the capacity of terrestrial ecosystems
#' to retain biological diversity under climate change: The bioclimatic
#' ecosystem resilience index. Ecological Indicators, 117, 106554.
#' \doi{10.1016/j.ecolind.2020.106554}. Introduces the BERI framework from
#' which the connectedness calculations are derived.
#' @export
#' @examples
#' condition_file <- system.file("extdata", "site_condition.tif", package = "connectivity")
#' pa_file <- system.file("extdata", "pa_proportion.tif", package = "connectivity")
#'
#' \donttest{
#' # Connectedness
#' conn <- connectedness(
#'     condition_file = condition_file,
#'     lambdas = c(2, 20, 200),
#'     max_cost = 2,
#'     window_size = 5,
#'     outer_window = 9,
#'     levels = c(2, 4, 8, 16, 32),
#'     window_mode = "circular",
#'     option = 1,
#'     n_threads = 2
#' )
#'
#' # Connected Habitat
#' habitat <- connectedness(
#'     condition_file = condition_file,
#'     lambdas = c(2, 20, 200),
#'     max_cost = 2,
#'     window_size = 5,
#'     outer_window = 9,
#'     levels = c(2, 4, 8, 16, 32),
#'     window_mode = "circular",
#'     option = 3,
#'     n_threads = 2
#' )
#'
#' # PARC-connectedness with default settings
#' parc <- connectedness(condition_file, pa_file = pa_file, n_threads = 2)
#'
#' terra::plot(c(conn, habitat, parc),
#'             main = c("Connectedness", "Connected habitat", "PARC-connectedness"))
#' }
connectedness <- function(condition_file,
                          pa_file = NULL,
                          pa_to_pa = FALSE,
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
                          option = 3L,
                          n_threads = NULL,
                          filename = "",
                          overwrite = FALSE,
                          wopt = list()) 
{
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
                                   settings, scales[[1L]])
    
    name <- if (option == 1L) "connectedness" else "connected_habitat"
    
    if (!is.null(pa_file)) {
        pa <- .read_window(pa_file, prepared, "pa_file", scale = scales[[2L]])
        .check_matrix(pa, "pa_file", unit_interval = TRUE)
        prepared$condition <- pmax(prepared$condition, pa)
        prepared$mask <- !(pa > 0 & !is.na(prepared$condition))
        prepared$mask[is.na(prepared$mask)] <- TRUE
        name <- "parc_connectedness"
    }
    
    traversal <- .make_traversal(prepared$condition, resistance_file, resistance_scale, prepared)
    score <- .run_connectivity(prepared, settings, traversal, pa_to_pa = pa_to_pa)
    
    if (is.null(pa_file) && option != 1L) {
        score <- .float32(score * prepared$condition)
        if (option == 3L) score <- .float32(sqrt(score))
    }
    
    return(
        .raster_result(
            score, prepared$template, name, filename, overwrite, wopt,
            prepared$polygons, prepared$output_extent
        )
    )
}
