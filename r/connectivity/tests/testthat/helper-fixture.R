# Same asymmetric, multi-level tile and frozen samples as the Rust public API
# and Python binding tests. R uses one-based indices below.
multilevel <- function(mode = "circular", beri = FALSE, pa_to_pa = FALSE) {
  i <- row(matrix(0, 13, 14)) - 1L
  j <- col(i) - 1L
  condition <- ((i * 7L + j * 13L) %% 10L + 1L) / 10
  condition[(i * 7L + j * 11L) %% 29L == 0L] <- NA_real_
  mask <- is.na(condition) | (i * 3L + j * 5L) %% 7L == 0L
  traversal <- ((i * 5L + j * 3L) %% 11L) / 10
  traversal[(i + j * 3L) %% 13L == 0L] <- NA_real_
  transgrids <- NULL
  if (beri) {
    transgrids <- lapply(0:1, function(scenario) {
      grid <- array(0, c(13, 14, 2))
      for (band in 0:1) {
        values <- ((i * 7L + j * 3L + scenario * 11L + band * 2L) %% 23L) / 23
        values[(i * 3L + j * 5L + scenario + band) %% 31L == 0L] <- NA_real_
        grid[, , band + 1L] <- values
      }
      grid
    })
  }
  list(condition = condition, mask = mask, traversal = traversal,
       transgrids = transgrids, transform = c(1000, 0, 500000, 0, -1000, 6000000),
       levels = c(1, 2, 4), lambdas = c(2, 20), max_cost = 20,
       window_size = 3L, outer_window = 5L, offsets = c(3L, 5L),
       window_mode = mode, pa_to_pa = pa_to_pa, n_threads = 1L)
}

fixture_cells <- rbind(c(1L, 2L), c(4L, 6L), c(7L, 9L), c(13L, 14L))
