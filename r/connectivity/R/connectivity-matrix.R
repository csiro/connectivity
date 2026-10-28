#' Calculate connectivity on an in-memory raster matrix
#'
#' Calls the same Rust numerical engine as the Python package. This
#' low-level interface returns raw connectedness, or BERI when `transgrids` are
#' supplied. It does not apply the Connected Habitat condition multiplier or
#' square root, read raster files, scale inputs, or plan tiles.
#'
#' @param condition Numeric matrix of habitat condition in `[0, 1]`, with rows
#'   running from top to bottom and columns from left to right. `NA` and `NaN`
#'   indicate missing cells. Inputs are converted to single precision in Rust.
#' @param transform Six numbers in rasterio/affine order `(a, b, c, d, e, f)`:
#'   `x = a * col + b * row + c`, `y = d * col + e * row + f`, using zero-based
#'   pixel corners. This is not GDAL geotransform order. Projected coordinates
#'   must be metres; geographic coordinates must be longitude/latitude degrees.
#'   For tiles, supply the transform of the full raster, together with `offsets`.
#' @param mask Logical matrix matching `condition`. `TRUE` skips a focal cell,
#'   but finite masked cells still contribute to paths and habitat. Missing
#'   condition cells are always masked. The mask cannot contain `NA`.
#' @param traversal Optional numeric matrix in `[0, 1]`, matching `condition`.
#'   High values mean easy passage, i.e. `1 - resistance`. Missing values fall
#'   back to condition in the core. Omit to use condition for path costs.
#' @param transgrids Optional non-empty list of numeric arrays with dimensions
#'   `(rows, columns, bands)`. Supply current climate first, then future
#'   scenarios. All arrays must have the same dimensions and band order.
#'   Missing bands give a cell climate similarity of zero.
#' @param levels Continuous powers of two for raster aggregation. Level 1 is
#'   added automatically; levels are sorted and deduplicated.
#' @param lambdas Positive dispersal bandwidths in kilometres. Scores are
#'   averaged across bandwidths.
#' @param is_geo Whether coordinates are longitude/latitude degrees.
#' @param max_cost Finite relative traversal cost, at least 1.
#' @param window_size Positive odd window size for non-coarsest levels.
#' @param outer_window Positive odd coarsest window size, at least `window_size`.
#' @param offsets Non-negative, zero-based `(row, column)` offset of a tile in
#'   the full raster. Defaults to `(0, 0)` for a whole raster.
#' @param window_mode Either `"circular"` (default) or `"square"`.
#' @param pa_to_pa Restrict destinations to protected cells, identified by
#'   `mask == FALSE`. This requires caller-prepared PARC condition and mask.
#'   Only supported without `transgrids`.
#' @param n_threads Optional positive integer number of worker threads.
#'   `NULL` uses Rayon's default thread count.
#' @return A numeric matrix with the dimensions and dimnames of `condition`.
#'   Masked cells are `NA`. Computation uses the core's single precision;
#'   results are returned as ordinary R doubles.
#' @noRd
connectivity_matrix <- function(
    condition, transform, mask = is.na(condition), traversal = NULL,
    transgrids = NULL, levels = c(1, 2, 4, 8, 16, 32),
    lambdas = c(2, 20, 200), is_geo = FALSE, max_cost = 2,
    window_size = 3L, outer_window = 9L, offsets = c(0L, 0L),
    window_mode = c("circular", "square"), pa_to_pa = FALSE,
    n_threads = NULL) {
  condition <- .check_matrix(condition, "condition", unit_interval = TRUE)
  shape <- dim(condition)
  if (!is.matrix(mask) || !is.logical(mask) || !identical(dim(mask), shape) ||
      anyNA(mask)) {
    stop("mask must be a logical matrix matching condition, without NA", call. = FALSE)
  }
  mask <- mask | is.na(condition)
  if (!is.null(traversal)) {
    traversal <- .check_matrix(traversal, "traversal", unit_interval = TRUE)
    if (!identical(dim(traversal), shape)) {
      stop("traversal must have the same dimensions as condition", call. = FALSE)
    }
  }
  bands <- 0L
  if (!is.null(transgrids)) {
    if (!is.list(transgrids) || !length(transgrids)) {
      stop("transgrids must be NULL or a non-empty list of 3D numeric arrays", call. = FALSE)
    }
    for (grid in transgrids) {
      dims <- dim(grid)
      if (!is.numeric(grid) || is.complex(grid) || length(dims) != 3L ||
          !identical(dims[1:2], shape) || dims[3] < 1L ||
          (bands != 0L && dims[3] != bands) || any(is.infinite(grid))) {
        stop("transgrids must be 3D numeric arrays matching condition and band count, without Inf", call. = FALSE)
      }
      bands <- dims[3]
    }
  }
  .check_numbers(transform, "transform", length = 6L)
  if (transform[1] * transform[5] - transform[2] * transform[4] == 0) {
    stop("transform must be invertible", call. = FALSE)
  }
  .check_integers(levels, "levels", minimum = 1L)
  levels <- sort(unique(c(1L, as.integer(levels))))
  powers <- log2(levels)
  if (any(powers != floor(powers)) || any(diff(powers) != 1)) {
    stop("levels must form a continuous sequence of powers of two", call. = FALSE)
  }
  .check_numbers(lambdas, "lambdas", minimum = 0, strict = TRUE)
  .check_numbers(max_cost, "max_cost", length = 1L, minimum = 1)
  .check_integers(window_size, "window_size", length = 1L, minimum = 1L)
  .check_integers(outer_window, "outer_window", length = 1L, minimum = 1L)
  if (window_size %% 2 != 1 || outer_window %% 2 != 1 || outer_window < window_size) {
    stop("window sizes must be odd and outer_window must be at least window_size", call. = FALSE)
  }
  .check_integers(offsets, "offsets", length = 2L, minimum = 0L)
  .check_flag(is_geo, "is_geo")
  .check_flag(pa_to_pa, "pa_to_pa")
  if (pa_to_pa && !is.null(transgrids)) {
    stop("pa_to_pa is only supported without transgrids", call. = FALSE)
  }
  if (!is.null(n_threads)) {
    .check_integers(n_threads, "n_threads", length = 1L, minimum = 1L)
  }
  window_mode <- match.arg(window_mode)
  result <- connectivity_matrix_impl(
    as.double(condition), as.integer(mask), shape, as.double(transform),
    as.double(traversal), lapply(transgrids, as.double), as.integer(bands),
    levels, as.double(lambdas), is_geo, as.double(max_cost),
    as.integer(window_size), as.integer(outer_window), as.integer(offsets),
    window_mode, pa_to_pa, if (is.null(n_threads)) 0L else as.integer(n_threads)
  )
  # Raise R errors only after the native call has returned and Rust-owned
  # resources have been dropped. The native result_list feature returns errors.
  if (!is.null(result$err)) stop(as.character(result$err), call. = FALSE)
  matrix(result$ok, nrow = shape[1], ncol = shape[2], dimnames = dimnames(condition))
}

.check_matrix <- function(x, name, unit_interval = FALSE) {
  if (!is.matrix(x) || !is.numeric(x) || is.complex(x) || any(dim(x) == 0L) ||
      any(is.infinite(x))) {
    stop(name, " must be a non-empty numeric matrix without Inf", call. = FALSE)
  }
  if (unit_interval && any(x < 0 | x > 1, na.rm = TRUE)) {
    stop(name, " values must be between 0 and 1 or missing", call. = FALSE)
  }
  x
}

.check_numbers <- function(x, name, length = NULL, minimum = -Inf, strict = FALSE) {
  if (!is.numeric(x) || is.complex(x) || !base::length(x) ||
      (!is.null(length) && base::length(x) != length) ||
      any(!is.finite(x)) || any(if (strict) x <= minimum else x < minimum)) {
    stop(name, " must contain finite numeric values of the required length and range", call. = FALSE)
  }
}

.check_integers <- function(x, name, length = NULL, minimum = 0L) {
  .check_numbers(x, name, length, minimum)
  if (any(x != floor(x)) || any(x > .Machine$integer.max)) {
    stop(name, " must contain representable R integers", call. = FALSE)
  }
}

.check_flag <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    stop(name, " must be TRUE or FALSE", call. = FALSE)
  }
}
