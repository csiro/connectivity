# Raster preparation stays in R; numerical aggregation and routing stay in Rust.
.as_raster <- function(x, name, single = TRUE, values = TRUE) {
    if (is.character(x) && length(x) == 1L && !is.na(x) && nzchar(x)) {
        x <- terra::rast(x)
    }
    if (!inherits(x, "SpatRaster")) {
        stop(name, " must be a SpatRaster or a raster filename", call. = FALSE)
    }
    if (any(terra::is.rotated(x))) {
        stop(name, " has a rotated grid; rectify it before analysis", call. = FALSE)
    }
    if (single && terra::nlyr(x) != 1L) {
        stop(name, " must have exactly one layer", call. = FALSE)
    }
    if (values && !terra::hasValues(x)) {
        stop(name, " has no raster values", call. = FALSE)
    }
    return(x)
}

.as_polygons <- function(x, reference, name = "polygon_mask") {
    if (!inherits(x, "SpatVector")) x <- terra::vect(x)
    # There are no coordinates to transform for an empty vector. terra also
    # drops its geometry type when all features are removed.
    if (!nrow(x)) return(x)
    if (nrow(x) && terra::geomtype(x) != "polygons") {
        stop(name, " must contain polygon geometries", call. = FALSE)
    }
    if (nrow(x) && !all(terra::is.valid(x))) {
        stop(name, " contains invalid polygons; repair them before analysis", call. = FALSE)
    }
    has_source <- nzchar(terra::crs(x))
    has_target <- nzchar(terra::crs(reference))
    if (has_source != has_target) {
        stop(name, " and the raster must both have a CRS, or both omit one", call. = FALSE)
    }
    if (has_source && !terra::same.crs(x, reference)) {
        x <- terra::project(x, terra::crs(reference))
    }
    return(x)
}

.affine <- function(raster) {
    return(c(
        terra::res(raster)[1], 0, terra::xmin(raster),
        0, -terra::res(raster)[2], terra::ymax(raster)
    ))
}

.coordinate_info <- function(raster) {
    transform <- .affine(raster)
    if (!nzchar(terra::crs(raster))) {
        resolution <- terra::res(raster)
        bounds <- as.vector(terra::ext(raster))
        geographic <- if (all(resolution < 1)) TRUE else if (all(resolution > 1)) FALSE else {
            all(bounds[1:2] >= -180 & bounds[1:2] <= 180) &&
                all(bounds[3:4] >= -90 & bounds[3:4] <= 90)
        }
        warning("Raster CRS is missing; assuming ",
                if (geographic) "longitude/latitude degrees" else "projected metres",
                call. = FALSE)
    } else {
        geographic <- terra::is.lonlat(raster)
        if (!geographic) {
            units <- terra::linearUnits(raster)
            if (!is.finite(units) || units <= 0) {
                stop("Cannot determine the projected CRS units in metres", call. = FALSE)
            }
            transform <- transform * units
        }
    }
    return(list(transform = transform, is_geo = geographic))
}

# Match the Python reader's float32 input conversion and arithmetic. R itself
# stores numeric vectors as doubles, so convert through a four-byte binary buffer.
.float32 <- function(x) {
    out <- readBin(writeBin(as.double(x), raw(), size = 4L), "double",
                   n = length(x), size = 4L)
    attributes(out) <- attributes(x)
    return(out)
}

.check_scale <- function(scale, name = "scale") {
    if (!is.null(scale)) .check_numbers(scale, name, length = 1L, minimum = 0)
    return(scale)
}

.condition_scales <- function(scale) {
    if (is.null(scale)) return(list(NULL, NULL))
    if ((!is.numeric(scale) && !is.list(scale)) || !length(scale) || length(scale) > 2L) {
        stop("scale must be a number or a pair of condition and protected-area scales", call. = FALSE)
    }
    scales <- as.list(scale)
    if (length(scales) == 1L) scales <- c(scales, list(NULL))
    return(lapply(scales, .check_scale))
}

.normalise_settings <- function(levels, lambdas, max_cost, window_size,
                                outer_window, window_mode, n_threads) {
    if (is.null(levels)) stop("levels must be provided", call. = FALSE)
    if (length(levels)) .check_integers(levels, "levels", minimum = 1L)
    lower <- 2^floor(log2(levels))
    rounded <- ifelse(levels - lower <= 2 * lower - levels, lower, 2 * lower)
    levels <- sort(unique(c(1, rounded)))
    .check_integers(levels, "levels", minimum = 1L)
    if (any(diff(log2(levels)) != 1)) {
        stop("levels must form a continuous power-of-two sequence", call. = FALSE)
    }
    .check_numbers(lambdas, "lambdas", minimum = 0, strict = TRUE)
    .check_numbers(max_cost, "max_cost", length = 1L, minimum = 1)
    .check_integers(window_size, "window_size", length = 1L, minimum = 1L)
    .check_integers(outer_window, "outer_window", length = 1L, minimum = 1L)
    if (outer_window < window_size) {
        message("outer_window increased to window_size (", window_size, ")")
        outer_window <- window_size
    }
    if (window_size %% 2 != 1 || outer_window %% 2 != 1) {
        stop("window sizes must be odd", call. = FALSE)
    }
    if (!is.character(window_mode) || length(window_mode) != 1L || is.na(window_mode)) {
        stop("window_mode must be 'circular' or 'square'", call. = FALSE)
    }
    window_mode <- tolower(trimws(window_mode))
    if (!window_mode %in% c("circular", "square")) {
        stop("window_mode must be 'circular' or 'square'", call. = FALSE)
    }
    if (!is.null(n_threads)) .check_integers(n_threads, "n_threads", length = 1L, minimum = 1L)
    return(list(
        levels = as.integer(levels), lambdas = lambdas, max_cost = max_cost,
        window_size = window_size, outer_window = outer_window,
        window_mode = window_mode, n_threads = n_threads
    ))
}

# Bounds are converted to half-open, zero-based pixel windows on the full grid.
.pixel_window <- function(bounds, raster, padding = 0, snap = 1) {
    resolution <- terra::res(raster)
    cols <- c(floor((bounds[1] - padding - terra::xmin(raster)) / resolution[1] + 1e-9),
              ceiling((bounds[2] + padding - terra::xmin(raster)) / resolution[1] - 1e-9))
    rows <- c(floor((terra::ymax(raster) - bounds[4] - padding) / resolution[2] + 1e-9),
              ceiling((terra::ymax(raster) - bounds[3] + padding) / resolution[2] - 1e-9))
    cols <- c(max(0, floor(cols[1] / snap) * snap),
              min(ncol(raster), ceiling(cols[2] / snap) * snap))
    rows <- c(max(0, floor(rows[1] / snap) * snap),
              min(nrow(raster), ceiling(rows[2] / snap) * snap))
    if (rows[2] <= rows[1] || cols[2] <= cols[1]) {
        stop("polygon_mask does not overlap the raster", call. = FALSE)
    }
    return(list(
        offsets = as.integer(c(rows[1], cols[1])),
        extent = terra::ext(c(
            terra::xmin(raster) + cols * resolution[1],
            rev(terra::ymax(raster) - rows * resolution[2])
        ))
    ))
}

.polygon_cells <- function(polygons, template) {
    covered <- terra::rasterize(polygons, terra::rast(template),
                                field = 1, background = 0, touches = TRUE)
    return(matrix(terra::as.array(covered)[, , 1L] != 0, nrow(template), ncol(template)))
}

# A pixel margin is planar even on a longitude/latitude grid. Reconstruct a
# temporary local geometry so terra does not interpret the width as metres on
# the ellipsoid. The caller's polygon and CRS are never modified.
.buffer_in_map_units <- function(polygons, width) {
    buffered <- terra::vect(terra::geom(polygons, wkt = TRUE), crs = "local")
    buffered <- terra::buffer(buffered, width = width)
    terra::crs(buffered) <- terra::crs(polygons)
    return(buffered)
}

.prepare_landscape <- function(condition_file, polygon_mask, closed_border,
                               margin_px, settings, scale) {
    reference <- .as_raster(condition_file, "condition_file")
    .check_flag(closed_border, "closed_border")
    .check_integers(margin_px, "margin_px", length = 1L, minimum = 0L)
    .check_scale(scale)
    coordinates <- .coordinate_info(reference)
    polygons <- NULL
    offsets <- c(0L, 0L)
    output_extent <- terra::ext(reference)
    template <- reference
    if (!is.null(polygon_mask)) {
        polygons <- .as_polygons(polygon_mask, reference)
        if (!nrow(polygons) || !nrow(terra::geom(polygons))) {
            stop("polygon_mask has no polygon area", call. = FALSE)
        }
        bounds <- as.vector(terra::ext(polygons))
        output_extent <- .pixel_window(bounds, reference)$extent
        padding <- 0
        if (!closed_border) {
            limit <- (settings$outer_window + 3) * max(settings$levels)
            if (margin_px > limit) {
                stop("margin_px cannot exceed the non-closed read-buffer limit (", limit, ")", call. = FALSE)
            }
            padding <- (limit + 0.5) * max(terra::res(reference))
        }
        window <- .pixel_window(bounds, reference, padding,
                                snap = if (closed_border) 1 else max(settings$levels))
        offsets <- window$offsets
        template <- terra::crop(reference, window$extent, snap = "near")
    }
    prepared <- list(reference = reference, template = template, polygons = polygons,
                     offsets = offsets, output_extent = output_extent,
                     transform = coordinates$transform, is_geo = coordinates$is_geo,
                     closed_border = closed_border)
    condition <- .read_window(reference, prepared, "condition_file", scale = scale)
    mask <- is.na(condition)
    if (!is.null(polygons)) {
        inside <- .polygon_cells(polygons, template)
        if (closed_border) {
            condition[!inside] <- NA_real_
            mask <- is.na(condition)
        } else {
            valid <- if (margin_px == 0) inside else {
                .polygon_cells(.buffer_in_map_units(polygons, margin_px * max(terra::res(reference))), template)
            }
            mask <- mask | !valid
        }
    }
    prepared$condition <- condition
    prepared$mask <- mask
    return(prepared)
}

.read_window <- function(x, prepared, name, scale = NULL, single = TRUE) {
    raster <- .as_raster(x, name, single = single)
    target <- prepared$template
    tolerance <- terra::res(target) * 1e-8
    if (!terra::same.crs(raster, target) ||
        any(abs(terra::res(raster) - terra::res(target)) > tolerance)) {
        stop(name, " must have the same CRS, resolution and grid alignment as condition_file", call. = FALSE)
    }
    cropped <- terra::crop(raster, terra::ext(target), snap = "near")
    if (!identical(c(nrow(cropped), ncol(cropped)), c(nrow(target), ncol(target))) ||
        any(abs(as.vector(terra::ext(cropped)) - as.vector(terra::ext(target))) > rep(tolerance, each = 2))) {
        stop(name, " must cover the analysis window on the same grid as condition_file", call. = FALSE)
    }
    values <- .float32(terra::as.array(cropped))
    if (single) values <- matrix(values[, , 1L], nrow(target), ncol(target))
    if (!is.null(scale) && !scale %in% c(0, 1)) values <- .float32(values / scale)
    if (prepared$closed_border && !is.null(prepared$polygons)) {
        outside <- !.polygon_cells(prepared$polygons, target)
        if (single) values[outside] <- NA_real_ else {
            for (band in seq_len(dim(values)[3])) {
                layer <- values[, , band, drop = FALSE]
                layer[outside] <- NA_real_
                values[, , band] <- layer
            }
        }
    }
    return(values)
}

.make_traversal <- function(condition, resistance_file, resistance_scale, prepared) {
    .check_scale(resistance_scale, "resistance_scale")
    if (is.null(resistance_file)) return(NULL)
    resistance <- .read_window(resistance_file, prepared, "resistance_file", scale = resistance_scale)
    if (any(is.infinite(resistance))) stop("resistance_file contains Inf", call. = FALSE)
    traversal <- .float32(1 - resistance)
    traversal[] <- pmin(1, pmax(0, traversal))
    gaps <- is.na(resistance) & !is.na(condition)
    if (any(gaps)) {
        warning(sum(gaps), " cell(s) have valid condition but missing resistance; using condition for path traversal there",
                call. = FALSE)
        traversal[gaps] <- condition[gaps]
    }
    traversal[is.na(condition)] <- NA_real_
    return(traversal)
}

.run_connectivity <- function(prepared, settings, traversal, transgrids = NULL, pa_to_pa = FALSE) {
    return(do.call(connectivity_matrix, c(list(
        condition = prepared$condition, transform = prepared$transform,
        mask = prepared$mask, traversal = traversal, transgrids = transgrids,
        is_geo = prepared$is_geo, offsets = prepared$offsets, pa_to_pa = pa_to_pa
    ), settings)))
}

.raster_result <- function(score, template, name, filename, overwrite, wopt,
                           polygons = NULL, output_extent = NULL) {
    if (!is.character(filename) || length(filename) != 1L || is.na(filename)) {
        stop("filename must be a single string", call. = FALSE)
    }
    .check_flag(overwrite, "overwrite")
    if (!is.list(wopt) || (length(wopt) && (is.null(names(wopt)) || any(!nzchar(names(wopt)))))) {
        stop("wopt must be a named list of terra write options", call. = FALSE)
    }
    out <- terra::rast(template[[1L]])
    terra::values(out) <- as.vector(t(score))
    names(out) <- name
    if (!is.null(polygons)) {
        out <- terra::crop(out, output_extent, snap = "near")
        out <- terra::mask(out, polygons, touches = TRUE)
    }
    if (nzchar(filename)) {
        options <- utils::modifyList(list(datatype = "FLT4S"), wopt)
        out <- terra::writeRaster(out, filename, overwrite = overwrite, wopt = options)
    }
    return(out)
}
