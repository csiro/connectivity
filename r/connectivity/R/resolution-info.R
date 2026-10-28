#' Preview raster aggregation levels and search reach
#'
#' A planning helper equivalent to the Python package's `resolution_info()`.
#' Reports raster metadata, candidate grid dimensions, resolutions, cell counts,
#' approximate search reach and small-grid flags without reading raster values.
#' The Rust engine generates levels internally; external overviews are unused.
#'
#' @param file_path Raster filename or `terra::SpatRaster`. Multiple layers and
#'   rasters without values are supported; only grid metadata are used.
#' @param levels Positive integer aggregation factors. `NULL` uses powers of two
#'   from 1 to 512. Candidates are sorted and deduplicated, without rounding or
#'   inserting intermediate levels. Indicator functions normalise levels separately.
#' @param min_dimension Integer small-grid threshold, clamped to at least 1.
#' @param outer_window Integer search-window size, clamped to at least 1.
#' @return Invisibly, a data frame with `level`, `width`, `height`, `cells`,
#'   `percent_base`, `x_resolution`, `y_resolution`, `reach`, `reach_km` and
#'   `note`. Resolutions and `reach` use native CRS units (also stored in the
#'   `unit` attribute). `reach_km` is `NA` when CRS units are unknown.
#' @details Dimensions round up after division by each candidate level. Search
#'   reach follows Python's planning convention:
#'   `outer_window * level * max(x_resolution, y_resolution)` using base
#'   resolutions. It is an approximate planning span, not a dispersal bandwidth.
#'   Geographic distances use 111.32 km per degree of latitude and its cosine
#'   adjustment at the raster's centre latitude for longitude. Projected units
#'   are converted using the CRS conversion factor. Small grids are labelled
#'   when either dimension is strictly below `min_dimension`.
#' @export
#' @examples
#' grid <- terra::rast(nrows = 65, ncols = 129, xmin = 0, xmax = 12900,
#'                     ymin = 0, ymax = 6500, crs = "EPSG:3857")
#' plan <- resolution_info(grid, levels = c(1, 2, 4, 8), outer_window = 7)
resolution_info <- function(file_path, levels = NULL, min_dimension = 16L,
                            outer_window = 5L) {
    if (is.null(levels)) levels <- 2^(0:9)
    .check_integers(levels, "levels", minimum = 1L)
    .check_integers(min_dimension, "min_dimension", length = 1L, minimum = -Inf)
    .check_integers(outer_window, "outer_window", length = 1L, minimum = -Inf)
    levels <- sort(unique(as.integer(levels)))
    min_dimension <- max(1, min_dimension)
    outer_window <- max(1, outer_window)
    raster <- .as_raster(file_path, "file_path", single = FALSE, values = FALSE)
    resolution <- abs(terra::res(raster))
    distances <- .resolution_distances(raster, resolution)
    width <- ceiling(ncol(raster) / levels)
    height <- ceiling(nrow(raster) / levels)
    cells <- width * height
    plan <- data.frame(
        level = levels, width = width, height = height, cells = cells,
        percent_base = 100 * cells / terra::ncell(raster),
        x_resolution = resolution[1] * levels,
        y_resolution = resolution[2] * levels,
        reach = outer_window * levels * max(resolution),
        reach_km = outer_window * levels * max(distances$km),
        note = ifelse(pmin(width, height) < min_dimension, "small grid; review", "")
    )
    attr(plan, "unit") <- distances$unit

    crs <- terra::crs(raster, describe = TRUE)
    crs_label <- crs$name[1]
    if (!is.na(crs$authority[1]) && !is.na(crs$code[1])) {
        crs_label <- paste0(crs$authority[1], ":", crs$code[1], " (", crs_label, ")")
    }
    datatype <- unique(terra::datatype(raster))
    datatype[!nzchar(datatype)] <- "unknown (in memory)"
    cat("\nFile: ", if (is.character(file_path)) file_path else "<SpatRaster>",
        "\nBase raster: ", ncol(raster), " x ", nrow(raster), " cells (",
        .format_compact(terra::ncell(raster)), " total)",
        "\nBase resolution: ", .format_resolution(resolution[1], resolution[2], distances$unit),
        "\nCRS: ", crs_label,
        "\nBands: ", terra::nlyr(raster),
        "\nData type(s): ", paste(datatype, collapse = ", "),
        "\nNodata: ", .resolution_nodata(raster),
        "\nOuter window: ", outer_window,
        "\nSmall-dimension warning threshold: < ", min_dimension, " cells\n", sep = "")
    if (!is.null(distances$note)) cat("Distance note: ", distances$note, "\n", sep = "")
    cat("\nCandidate internally generated levels:\n")
    display <- data.frame(
        Level = levels, Width = width, Height = height,
        Cells = .format_compact(cells),
        `% base` = sprintf("%.3f%%", plan$percent_base),
        Resolution = .format_resolution(plan$x_resolution, plan$y_resolution, distances$unit),
        Reach = paste(.format_compact(plan$reach), distances$unit),
        `Reach km` = ifelse(is.na(plan$reach_km), "-", paste(.format_compact(plan$reach_km), "km")),
        Note = plan$note, check.names = FALSE
    )
    columns <- lapply(names(display), function(name) {
        return(format(c(name, as.character(display[[name]])), justify = "right"))
    })
    cat(do.call(paste, c(columns, sep = " ")), sep = "\n")
    return(invisible(plan))
}

.resolution_nodata <- function(raster) {
    flags <- terra::NAflag(raster)
    if (any(!is.na(flags))) return(paste(flags, collapse = ", "))
    sources <- unique(terra::sources(raster))
    sources <- sources[nzchar(sources)]
    if (!length(sources)) return("NA (in memory)")
    # NAflag reports terra overrides, not the source file's native nodata value.
    nodata <- lapply(sources, function(source) {
        description <- terra::describe(source, options = c("nomd", "norat", "noct", "nofl"))
        values <- sub("^.*NoData Value=", "", description[grepl("NoData Value=", description)])
        return(if (length(values)) trimws(values) else "none")
    })
    return(paste(unique(unlist(nodata)), collapse = ", "))
}

.resolution_distances <- function(raster, resolution) {
    unit <- "map units"
    km <- c(NA_real_, NA_real_)
    note <- NULL
    crs <- terra::crs(raster)
    if (nzchar(crs)) {
        if (terra::is.lonlat(raster)) {
            unit <- "degrees"
            latitude <- (terra::ymin(raster) + terra::ymax(raster)) / 2
            km <- resolution * 111.32 * c(abs(cos(latitude * pi / 180)), 1)
            note <- sprintf("approximate km values use centre latitude %.3f", latitude)
        } else {
            # terra returns WKT2; the final LENGTHUNIT describes projected axes.
            units <- regmatches(crs, gregexpr('LENGTHUNIT[[]"[^"]+"', crs))[[1]]
            if (length(units)) unit <- sub('LENGTHUNIT[[]"([^"]+)"', '\\1', units[length(units)])
            factor <- terra::linearUnits(raster)
            if (is.finite(factor) && factor > 0) km <- resolution * factor / 1000
        }
    }
    return(list(unit = unit, km = km, note = note))
}

.format_compact <- function(x) {
    return(vapply(x, function(value) {
        if (is.na(value)) return("-")
        if (value >= 1e9) return(sprintf("%.2fB", value / 1e9))
        if (value >= 1e6) return(sprintf("%.2fM", value / 1e6))
        if (value >= 1e3) return(sprintf("%.2fK", value / 1e3))
        if (value == floor(value)) return(sprintf("%.0f", value))
        return(sprintf("%.4g", value))
    }, character(1), USE.NAMES = FALSE))
}

.format_resolution <- function(x, y, unit) {
    equal <- abs(x - y) <= 1e-8 + 1e-5 * abs(y)
    return(paste(ifelse(equal, .format_compact(x),
                        paste(.format_compact(x), "x", .format_compact(y))), unit))
}
