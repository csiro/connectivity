#' Calculate habitat connectedness, Connected Habitat or PARC-connectedness
#'
#' @description
#' Habitat connectedness measures how much of the potentially connected habitat
#' around each cell is retained, relative to a hypothetical intact landscape.
#' Surrounding habitat counts more when it is in better condition and closer
#' along its least-cost path, which lengthens through degraded habitat. Values
#' range from 0 (isolated) to 1 (as connected as an intact landscape).
#'
#' By default, the function returns Connected Habitat (connectivity-adjusted
#' condition): the geometric mean of a cell's own habitat condition and its
#' connectedness, used as an input to the Biodiversity Habitat Index (BHI).
#' With `pa_file`, it returns the Protected Area
#' Connectedness Index (PARC-connectedness) instead: how well protected cells
#' are connected to other protected areas and to surrounding intact habitat.
#'
#' Each cell's neighbourhood keeps fine detail nearby and represents more
#' distant landscape with progressively coarser aggregated cells (Valavi et al.
#' 2026). Inputs are raster filenames or `terra` SpatRaster objects, and the
#' result is a georeferenced raster.
#'
#' @param condition_file Single-layer `terra::SpatRaster` or raster filename,
#'   or a multi-layer one with `band`. Habitat condition must be in `[0, 1]`
#'   after scaling. Missing values are excluded. All raster inputs must share a
#'   CRS, resolution and pixel grid; auxiliary rasters must cover the analysis
#'   window. They are not resampled.
#' @param ... Not used. Every argument after `condition_file` must be named,
#'   in full; an unnamed or unknown argument is an error.
#' @param band Optional layer of `condition_file` to use, numbered from 1 as in
#'   GDAL, rasterio and terra, for a filename or a `SpatRaster` alike: for
#'   example, one year of a condition time series. It selects a layer of
#'   `condition_file` only: `pa_file` and `resistance_file` must have one
#'   layer. `NULL` (default) requires a single-layer `condition_file`.
#' @param scale Optional divisor for condition values (`NULL`, 0 or 1 leaves
#'   values unchanged). A numeric pair or list supplies separate condition and
#'   protected-area divisors, e.g. `c(100, 100)` or `list(NULL, 100)`.
#' @param pa_file Optional single-layer protected-area proportion raster or
#'   filename. Values must be in `[0, 1]`
#'   after scaling. The analysis uses the maximum of condition and protected
#'   fraction and evaluates protected cells only. PARC output is raw
#'   connectedness, independent of `option`.
#' @param pa_to_pa Restrict PARC destinations to protected cells, weighted by
#'   protected fraction at coarser levels. Ignored if `pa_file` is `NULL`.
#' @param option Habitat output: 1 returns connectedness, 2 multiplies by
#'   condition, and 3 (default) takes the square root of that product.
#' @param resistance_file Optional single-layer resistance raster or filename.
#'   After scaling, `1 - resistance` is
#'   clipped to `[0, 1]` for path traversal, so resistance of 1 or more,
#'   including `Inf`, gives the highest path cost. Missing resistance at valid
#'   condition cells falls back to condition and raises a warning. Resistance
#'   does not alter habitat values or the domain.
#' @param resistance_scale Optional divisor applied only to resistance values.
#' @param lambdas Positive dispersal bandwidths in kilometres. Scores are
#'   averaged across bandwidths.
#' @param max_cost Finite relative traversal cost, at least 1.
#' @param levels Aggregation levels. Positive integers are rounded to the
#'   nearest power of two, with ties rounded down; level 1 is included. The
#'   result must be a continuous power-of-two sequence. Aggregated grids are
#'   computed from the base-resolution values.
#' @param window_size Positive odd window size for non-coarsest levels.
#' @param outer_window Positive odd coarsest window size, at least `window_size`.
#' @param window_mode Either `"circular"` (default) or `"square"`.
#' @param polygon_mask Optional polygon `terra::SpatVector`, vector filename or
#'   object accepted by [terra::vect()]. Polygons are reprojected to the raster
#'   CRS. Both polygon and raster must have a CRS, or both omit one. The result
#'   is cropped to the intersecting polygon bounds and cells outside polygons
#'   are `NA`. Cells touched by a polygon are included.
#' @param closed_border If `FALSE`, retain surrounding landscape context for
#'   the analysis before cropping the output. If `TRUE`, exclude the landscape
#'   outside `polygon_mask` from both paths and habitat contributions.
#' @param nodata_crossing How NoData cells of `condition_file` (sea, lakes,
#'   rivers) are treated. NoData in gaps at most `crossing_width` kilometres
#'   wide may be crossed, at `crossing_cost`; crossed NoData carries movement
#'   only, with no habitat area, protected area or climate. Wider gaps are not
#'   crossed.
#'   * `"strict"` (default): no wider gap is crossed anywhere, at any level,
#'     including rivers and inlets inside one landmass; routes around them
#'     within the search window are kept.
#'   * `"landmass"`: separate landmasses are linked only across such gaps;
#'     within one landmass coarse levels still link land across wider water
#'     (an inlet, or a river whose banks join), as with `NULL`.
#'   * `NULL`: the legacy handling of version 0.1.0, which ignores
#'     `crossing_width` and `crossing_cost`. The finest level does not cross
#'     NoData, but coarser levels link land on both sides of gaps narrower than
#'     about twice their cell size, at no cost, and only from some cells: a
#'     compact island near other land can have a linked part and a sealed
#'     part. Pass `nodata_crossing = NULL` to reproduce results of 0.1.0.
#'
#'   Without NoData every setting gives the same results. `"strict"` with
#'   `crossing_width = 0` crosses no gap, but it is not `NULL`: coarse levels
#'   never link land across NoData.
#' @param crossing_width Widest NoData gap crossed, in kilometres (finite,
#'   at least 0); 0 crosses none. The default, 0.5, is a provisional
#'   general-purpose assumption that allows limited crossing of narrow gaps,
#'   not a calibrated ecological threshold; wider gaps are sensitivity
#'   scenarios to choose explicitly. A gap's width is measured once at base
#'   resolution, as the widest disc that fits between the land cells; on
#'   geographic grids cells are narrower towards the poles. This also makes
#'   NoData in concave corners of a coastline crossable, within about a fifth
#'   of the width of the corner. With an open-border `polygon_mask`, the read
#'   window grows by this width. Ignored when `nodata_crossing` is `NULL`, with
#'   a message when changed.
#' @param crossing_cost Movement cost of crossed NoData per unit distance,
#'   relative to intact land (finite, at least 1): 1 costs as much as intact
#'   land, the default 4 four times as much. It is independent of condition,
#'   resistance and `max_cost`, and may exceed `max_cost`; a coarse cell
#'   holding land and crossable NoData costs the mean of their costs. Like the
#'   width, the default is a provisional assumption, not a calibrated value. No
#'   effect when `crossing_width` is 0; ignored when `nodata_crossing` is
#'   `NULL`, with a message when changed.
#' @param n_threads Optional positive integer number of worker threads.
#'   `NULL` uses Rayon's default thread count.
#' @param filename Optional output filename. An empty string returns an
#'   in-memory raster. Files default to single-precision floating-point values.
#' @param overwrite Whether an existing output file may be overwritten.
#' @param wopt Named list of [terra::writeRaster()] options.
#' @param save_log Whether to write a run log beside `filename` once it is
#'   written: a small text file of the settings, inputs and output of the run,
#'   with the output's extension replaced by `.toml` (`result.tif` gets
#'   `result.toml`). Eco-Connectivity Studio restores the settings of a log
#'   dropped on its window. An existing run log is replaced, but another file
#'   of that name is kept, with a warning. Not writing the log is a warning,
#'   never an error, and nothing is written when `filename` is empty. The
#'   default is the `connectivity.save_log` option, or when it is unset, `TRUE`
#'   unless the environment variable `CONNECTIVITY_SAVE_LOG` is `0`, `false`,
#'   `no` or `off`, which also turns off the Python package's logs.
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
#'   `(outer_window + 3) * max(levels)` pixels, plus `crossing_width` with a
#'   NoData crossing mode, and aligned to the full raster's coarsest
#'   aggregation grid, so merged tiles equal a whole-raster result.
#'
#'   After version 0.1.0 the default NoData handling changed from the legacy
#'   one to `nodata_crossing = "strict"`, so results change wherever NoData
#'   lies near land. Pass `nodata_crossing = NULL` to reproduce results of 0.1.0.
#' @seealso [make_tiles()] to split large rasters into tiles.
#' @references
#' Valavi, R., Mokany, K., Ware, C., Vickers, M., Giljohann, K. M., and
#' Ferrier, S. (2026). A scalable multi-resolution framework for
#' connectivity-based biodiversity indicators. EcoEvoRxiv, preprint.
#' \doi{10.32942/X2S68V}. Describes the multi-resolution implementation.
#'
#' Drielsma, M., Ferrier, S., and Manion, G. (2007). A raster-based technique
#' for analysing habitat configuration: The cost--benefit approach. Ecological
#' Modelling, 202(3--4), 324--332. \doi{10.1016/j.ecolmodel.2006.10.016}.
#' Introduces the raster cost-benefit approach to habitat connectedness and
#' distance-dependent aggregation.
#'
#' Drielsma, M. J., Love, J., Taylor, S., Thapa, R., and Williams, K. J.
#' (2022). General Landscape Connectivity Model (GLCM): a new way to map whole
#' of landscape biodiversity functional connectivity for operational planning
#' and reporting. Ecological Modelling, 465, 109858.
#' \doi{10.1016/j.ecolmodel.2021.109858}. Extends the approach to
#' whole-landscape connectivity mapping across large regions.
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
#'     nodata_crossing = "strict",
#'     crossing_width = 0.5,
#'     crossing_cost = 4,
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
#'     nodata_crossing = "strict",
#'     crossing_width = 0.5,
#'     crossing_cost = 4,
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
connectedness <- function(condition_file, ...,
                          band = NULL,
                          scale = NULL,
                          pa_file = NULL,
                          pa_to_pa = FALSE,
                          option = 3L,
                          resistance_file = NULL,
                          resistance_scale = NULL,
                          lambdas = c(2, 20, 200),
                          max_cost = 2,
                          levels = c(2, 4, 8, 16, 32),
                          window_size = 5L,
                          outer_window = 9L,
                          window_mode = "circular",
                          polygon_mask = NULL,
                          closed_border = FALSE,
                          nodata_crossing = "strict",
                          crossing_width = 0.5,
                          crossing_cost = 4,
                          n_threads = NULL,
                          filename = "",
                          overwrite = FALSE,
                          wopt = list(),
                          save_log = getOption("connectivity.save_log", .log_by_env()))
{
    started <- proc.time()[["elapsed"]]
    .check_named(...)
    .check_flag(save_log, "save_log")
    settings <- .normalise_settings(levels, lambdas, max_cost, window_size,
                                    outer_window, window_mode, n_threads,
                                    nodata_crossing, crossing_width, crossing_cost)
    .check_integers(option, "option", length = 1L, minimum = 1L)
    
    if (!option %in% 1:3) stop("option must be one of 1, 2 or 3", call. = FALSE)
    .check_flag(pa_to_pa, "pa_to_pa")
    if (pa_to_pa && is.null(pa_file)) {
        message("pa_to_pa is ignored because pa_file was not supplied")
        pa_to_pa <- FALSE
    }
    
    scales <- .condition_scales(scale)
    prepared <- .prepare_landscape(condition_file, polygon_mask, closed_border,
                                   settings, scales[[1L]], band)
    
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
    
    out <- .raster_result(
        score, prepared$template, name, filename, overwrite, wopt,
        prepared$polygons, prepared$output_extent
    )
    .save_log(
        filename, save_log, started,
        indicator = if (is.null(pa_file)) "connected_habitat" else "parc",
        settings = settings, option = option, pa_to_pa = pa_to_pa,
        condition_scale = scales[[1L]], protected_areas_scale = scales[[2L]],
        resistance_scale = resistance_scale, closed_border = closed_border,
        condition = .log_condition(condition_file, prepared$reference, band),
        resistance = if (!is.null(resistance_file)) .log_input(resistance_file),
        protected_areas = if (!is.null(pa_file)) .log_input(pa_file),
        polygon_mask = polygon_mask
    )
    return(out)
}
