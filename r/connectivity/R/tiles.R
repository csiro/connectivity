#' Create aggregation-aligned raster tiles
#'
#' Splits a raster grid into non-overlapping rectangular tile polygons for
#' processing large rasters piecewise. Pass each tile as `polygon_mask` to
#' [connectedness()] or [beri()] with open borders (the default), then merge the
#' results with [terra::merge()]. Open-border tiles read their surrounding
#' landscape, so the merged result matches a single whole-raster calculation.
#'
#' @param raster_file Reference `terra::SpatRaster` or raster filename. Only its
#'   grid is used, unless `balanced = TRUE`, which also reads the first layer's
#'   missing values.
#' @param nrows,ncols Positive integer numbers of tile rows and columns.
#' @param align_to Positive integer. Internal tile edges are placed at
#'   multiples of `align_to` pixels from the raster origin; the outer raster
#'   edge is always kept. Use the largest aggregation level, e.g. 32.
#' @param balanced If `FALSE`, tiles have near-equal sizes. If `TRUE`, row
#'   breaks, then the column breaks of each tile row, balance an estimated cost.
#'   Each `align_to` block costs its cell count if any of its cells in the first
#'   layer has a value, and nothing if all are missing, plus `io_weight` times
#'   its cell count.
#' @param io_weight Non-negative cost per cell added in balanced mode, so that
#'   mostly empty areas still carry some reading cost.
#' @param tile_id One-based tile number, in row-major order.
#' @return A `terra::SpatVector` of `nrows * ncols` polygons (one polygon for
#'   `make_tile()`) in the raster's CRS, ordered by row and then column, with
#'   attributes `id`, `row`, `col`, `row_start`, `row_stop`, `col_start`,
#'   `col_stop`, `align_to` and, in balanced mode, `cost`. Tile numbers and
#'   cell ranges are one-based and inclusive.
#' @seealso [connectedness()] and [beri()] for open-border polygon analyses.
#' @export
#' @examples
#' habitat <- terra::rast(nrows = 64, ncols = 96, xmin = 0, xmax = 9600,
#'                        ymin = 0, ymax = 6400, crs = "EPSG:3857")
#' terra::values(habitat) <- seq(0.1, 1, length.out = terra::ncell(habitat))
#' tiles <- make_tiles(habitat, nrows = 2, ncols = 2, align_to = 4)
#' parts <- lapply(seq_len(nrow(tiles)), function(i) {
#'     return(connectedness(habitat, polygon_mask = tiles[i], levels = c(2, 4),
#'                          outer_window = 5, n_threads = 1))
#' })
#' mosaic <- terra::merge(terra::sprc(parts))
make_tiles <- function(raster_file, nrows, ncols, align_to = 1L,
                       balanced = FALSE, io_weight = 0.1) {
    .check_integers(nrows, "nrows", length = 1L, minimum = 1L)
    .check_integers(ncols, "ncols", length = 1L, minimum = 1L)
    .check_integers(align_to, "align_to", length = 1L, minimum = 1L)
    .check_flag(balanced, "balanced")
    .check_numbers(io_weight, "io_weight", length = 1L, minimum = 0)
    raster <- .as_raster(raster_file, "raster_file", single = FALSE, values = balanced)
    height <- nrow(raster)
    width <- ncol(raster)
    costs <- NULL
    if (balanced) {
        blocks <- .block_costs(raster, align_to, io_weight)
        row_breaks <- .balanced_breaks(rowSums(blocks), nrows, align_to, height)
        col_breaks <- vector("list", nrows)
        costs <- matrix(0, nrows, ncols)
        for (row in seq_len(nrows)) {
            rows <- .break_blocks(row_breaks, row, align_to)
            col_breaks[[row]] <- .balanced_breaks(colSums(blocks[rows, , drop = FALSE]),
                                                  ncols, align_to, width)
            for (col in seq_len(ncols)) {
                cols <- .break_blocks(col_breaks[[row]], col, align_to)
                costs[row, col] <- sum(blocks[rows, cols])
            }
        }
    } else {
        row_breaks <- .aligned_breaks(height, nrows, align_to)
        col_breaks <- rep(list(.aligned_breaks(width, ncols, align_to)), nrows)
    }
    return(.tile_polygons(raster, row_breaks, col_breaks, align_to, costs))
}

#' @rdname make_tiles
#' @export
make_tile <- function(raster_file, nrows, ncols, tile_id, align_to = 1L,
                      balanced = FALSE, io_weight = 0.1) {
    .check_integers(nrows, "nrows", length = 1L, minimum = 1L)
    .check_integers(ncols, "ncols", length = 1L, minimum = 1L)
    .check_integers(tile_id, "tile_id", length = 1L, minimum = 1L)
    if (tile_id > nrows * ncols) {
        stop("tile_id must be between 1 and ", nrows * ncols, call. = FALSE)
    }
    tiles <- make_tiles(raster_file, nrows, ncols, align_to, balanced, io_weight)
    return(tiles[tile_id])
}

# Breaks are zero-based pixel offsets.
.aligned_breaks <- function(size, parts, align_to) {
    if (parts == 1L) return(c(0, size))
    internal <- (size - 1) %/% align_to
    if (internal < parts - 1) {
        stop("Cannot split ", size, " pixels into ", parts,
             " non-empty parts aligned to ", align_to, " pixels", call. = FALSE)
    }
    breaks <- 0
    for (k in seq_len(parts - 1L)) {
        lower <- breaks[k] %/% align_to + 1
        upper <- internal - (parts - k - 1)
        target <- floor(k * size / (parts * align_to) + 0.5)
        breaks <- c(breaks, min(max(target, lower), upper) * align_to)
    }
    return(c(breaks, size))
}

.balanced_breaks <- function(costs, parts, align_to, size) {
    blocks <- length(costs)
    if (blocks < parts) {
        stop("Cannot split ", blocks, " aligned blocks into ", parts,
             " non-empty parts", call. = FALSE)
    }
    total <- sum(costs)
    if (total <= 0) return(.aligned_breaks(size, parts, align_to))
    cumulative <- cumsum(costs)
    breaks <- 0
    previous <- 0
    for (k in seq_len(parts - 1L)) {
        target <- total * k / parts
        candidate <- sum(cumulative < target) + 1
        options <- if (candidate > 1) c(candidate, candidate - 1) else candidate
        options <- pmin(pmax(options, previous + 1), blocks - (parts - k))
        # which.min keeps the first option on ties.
        block <- options[which.min(abs(cumulative[options] - target))]
        breaks <- c(breaks, min(block * align_to, size))
        previous <- block
    }
    return(c(breaks, size))
}

.break_blocks <- function(breaks, part, align_to) {
    return((breaks[part] %/% align_to + 1):ceiling(breaks[part + 1L] / align_to))
}

.block_costs <- function(raster, align_to, io_weight) {
    valid <- !is.na(raster[[1L]])
    # Python's GDAL mask read marks a block valid when any of its cells is valid.
    if (align_to > 1L) valid <- terra::aggregate(valid, fact = align_to, fun = "max", na.rm = TRUE)
    sizes <- function(n) return(pmin(align_to, n - align_to * (seq_len(ceiling(n / align_to)) - 1)))
    area <- outer(sizes(nrow(raster)), sizes(ncol(raster)))
    return(area * (terra::as.matrix(valid, wide = TRUE) + io_weight))
}

.tile_polygons <- function(raster, row_breaks, col_breaks, align_to, costs) {
    nrows <- length(row_breaks) - 1L
    ncols <- length(col_breaks[[1L]]) - 1L
    tiles <- data.frame(id = seq_len(nrows * ncols), row = rep(seq_len(nrows), each = ncols),
                        col = rep(seq_len(ncols), times = nrows))
    tiles$row_start <- row_breaks[tiles$row] + 1
    tiles$row_stop <- row_breaks[tiles$row + 1L]
    tiles$col_start <- mapply(function(row, col) return(col_breaks[[row]][col] + 1), tiles$row, tiles$col)
    tiles$col_stop <- mapply(function(row, col) return(col_breaks[[row]][col + 1L]), tiles$row, tiles$col)
    tiles$align_to <- as.integer(align_to)
    if (!is.null(costs)) tiles$cost <- costs[cbind(tiles$row, tiles$col)]
    resolution <- terra::res(raster)
    left <- terra::xmin(raster) + (tiles$col_start - 1) * resolution[1]
    right <- terra::xmin(raster) + tiles$col_stop * resolution[1]
    top <- terra::ymax(raster) - (tiles$row_start - 1) * resolution[2]
    bottom <- terra::ymax(raster) - tiles$row_stop * resolution[2]
    corners <- cbind(
        id = rep(tiles$id, each = 5L), part = 1L,
        x = as.vector(rbind(left, right, right, left, left)),
        y = as.vector(rbind(bottom, bottom, top, top, bottom))
    )
    integers <- c("row_start", "row_stop", "col_start", "col_stop")
    tiles[integers] <- lapply(tiles[integers], as.integer)
    return(terra::vect(corners, type = "polygons", atts = tiles, crs = terra::crs(raster)))
}
